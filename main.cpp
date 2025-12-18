
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp> // 用glm处理向量/矩阵（需链接glm库）
#include <iostream>
#include <vector>

// ===================== 数据结构定义 =====================
// 形状类型枚举
enum ShapeType : int
{
    SHAPE_CIRCLE = 0,
    SHAPE_RECT = 1,
    SHAPE_SECTOR = 2
};

// 单个形状的风场参数
struct WindShape
{
    ShapeType type; // 形状类型
    int padding0;
    glm::vec2 pos;     // 中心位置 (x,y)
    glm::vec2 size;    // 尺寸：圆形(r,0)、矩形(w,h)、扇形(r,0)
    float rotation;    // 旋转角度（度）：矩形朝向/扇形起始角度
    float angleRange;  // 扇形终止角度-起始角度（仅扇形有效）
    glm::vec2 windDir; // 风向（归一化向量）
    float windSpeed;   // 风速（向量幅值）
    float padding1;
};

// 风场全局参数（传递到Shader）
struct WindFieldParams
{
    int shapeCount; // 形状数量
    int rtWidth;    // RT宽度
    int rtHeight;   // RT高度
    int padding1;
    WindShape shapes[128]; // 最多128个形状（可扩展）
};

// ===================== 全局变量 =====================
const int RT_WIDTH = 1024;
const int RT_HEIGHT = 768;
GLuint computeProgram;      // Compute Shader程序
GLuint windRT;              // 风场RT（存储向量：RG=xy分量，BA=预留）
GLuint uboParams;           // 风场参数UBO
WindFieldParams windParams; // 风场参数

// ===================== Shader编译 =====================
GLuint createComputeShader(const char* source)
{
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    // 检查编译错误
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "Compute Shader编译失败:\n" << infoLog << std::endl;
    }
    return shader;
}

// ===================== 初始化风场RT =====================
void initWindRT()
{
    glGenTextures(1, &windRT);
    glBindTexture(GL_TEXTURE_2D, windRT);
    // 用RGBA32F存储向量（RG=风向xy，精度足够）
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, RT_WIDTH, RT_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
}

// ===================== 初始化UBO =====================
void initUBO()
{
    glGenBuffers(1, &uboParams);
    glBindBuffer(GL_UNIFORM_BUFFER, uboParams);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(WindFieldParams), &windParams, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, uboParams); // 绑定到binding=0
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

// ===================== 初始化Compute Shader =====================
void initComputeShader()
{
    const char* csSource = R"(
        #version 430 core

        // 形状类型枚举（与CPU端一致）
        const int SHAPE_CIRCLE = 0;
        const int SHAPE_RECT = 1;
        const int SHAPE_SECTOR = 2;

        // 单个形状参数（与CPU端struct对齐）
        struct WindShape {
            int type;           // 形状类型（int占4字节，匹配CPU端enum）
            float padding0;     // 对齐：int(4) + padding(4) = 8字节
            vec2 pos;           // 中心位置 (8字节)
            vec2 size;          // 尺寸 (8字节)
            float rotation;     // 旋转角度（度）(4字节)
            float angleRange;   // 扇形角度范围 (4字节)
            vec2 windDir;       // 风向（归一化）(8字节)
            float windSpeed;    // 风速 (4字节)
            float padding1;
        };

        // 风场全局参数UBO
        layout(std140, binding = 0) uniform WindFieldParams {
            int shapeCount;     // 形状数量 (4字节)
            int rtWidth;        // RT宽度 (4字节)
            int rtHeight;       // RT高度 (4字节)
            int padding1;       // 对齐 (4字节)
            WindShape shapes[128]; // 形状数组
        } params;

        // 输出RT：RG=风向向量xy，BA=预留（0,0）
        layout(rgba32f, binding = 1) writeonly uniform image2D windRT;

        // 线程分组：16x16（适配GPU warp大小）
        layout(local_size_x = 16, local_size_y = 16) in;

        // ===================== 工具函数 =====================
        // 角度转弧度
        float deg2rad(float deg) {
            return deg * 3.1415926535 / 180.0;
        }

        // 旋转向量（绕原点，逆时针）
        vec2 rotateVec(vec2 v, float rad) {
            float c = cos(rad);
            float s = sin(rad);
            return vec2(v.x * c - v.y * s, v.x * s + v.y * c);
        }

        // 判定像素是否在圆形内
        bool isInCircle(vec2 pixelPos, WindShape shape) {
            float r = shape.size.x;
            float dist = length(pixelPos - shape.pos);
            return dist <= r;
        }

        // 判定像素是否在旋转矩形内
        bool isInRect(vec2 pixelPos, WindShape shape) {
            vec2 halfSize = shape.size * 0.5;
            vec2 delta = pixelPos - shape.pos;
            // 将像素相对坐标旋转（反向旋转，抵消矩形旋转）
            float rad = deg2rad(-shape.rotation);
            delta = rotateVec(delta, rad);
            // 判定是否在轴对齐矩形内
            return abs(delta.x) <= halfSize.x && abs(delta.y) <= halfSize.y;
        }

        // 判定像素是否在扇形内
        bool isInSector(vec2 pixelPos, WindShape shape) {
            float r = shape.size.x;
            vec2 delta = pixelPos - shape.pos;
            float dist = length(delta);
            if (dist > r) return false; // 超出半径

            // 计算像素相对于扇形中心的角度（[0, 360)）
            float angle = atan(delta.y, delta.x) * 180.0 / 3.1415926535;
            if (angle < 0.0) angle += 360.0;

            // 扇形起始/终止角度
            float startAngle = shape.rotation;
            float endAngle = startAngle + shape.angleRange;
            // 处理跨360°的情况
            if (endAngle > 360.0) {
                return angle >= startAngle || angle <= (endAngle - 360.0);
            } else {
                return angle >= startAngle && angle <= endAngle;
            }
        }

        // 计算形状对像素的风向向量（带衰减）
        vec2 getShapeWindVec(vec2 pixelPos, WindShape shape) {
            // 基础风向向量 = 方向 × 风速
            vec2 baseVec = shape.windDir * shape.windSpeed;

            return baseVec;
        }

        // ===================== 主逻辑 =====================
        void main() {
            // 获取当前线程对应的像素坐标
            ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);
            vec2 pixelPos = vec2(pixelCoord.x, pixelCoord.y);

            // 超出RT范围则返回
            if (pixelCoord.x >= params.rtWidth || pixelCoord.y >= params.rtHeight) {
                return;
            }

            // 初始化总风向向量为0
            vec2 totalWindVec = vec2(0.0);

            // 遍历所有形状，叠加风向
            for (int i = 0; i < params.shapeCount; i++) {
                WindShape shape = params.shapes[i];
                bool isInShape = false;

                // 判定像素是否在当前形状内
                switch (shape.type) {
                    case SHAPE_CIRCLE:
                        isInShape = isInCircle(pixelPos, shape);
                        break;
                    case SHAPE_RECT:
                        isInShape = isInRect(pixelPos, shape);
                        break;
                    case SHAPE_SECTOR:
                        isInShape = isInSector(pixelPos, shape);
                        break;
                    default:
                        isInShape = false;
                }

                // 若在形状内，叠加风向向量
                if (isInShape) {
                    totalWindVec += getShapeWindVec(pixelPos, shape);
                }
            }

            // 写入RT：RG=向量xy，BA=0（预留）
            imageStore(windRT, pixelCoord, vec4(totalWindVec, 0.0, 0.0));
        }
    )";

    // 创建并链接Compute Shader
    GLuint cs = createComputeShader(csSource);
    computeProgram = glCreateProgram();
    glAttachShader(computeProgram, cs);
    glLinkProgram(computeProgram);

    // 检查链接错误
    int success;
    char infoLog[512];
    glGetProgramiv(computeProgram, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(computeProgram, 512, NULL, infoLog);
        std::cerr << "Compute Program链接失败:\n" << infoLog << std::endl;
    }

    glDeleteShader(cs); // 链接后删除Shader
}

// ===================== 可视化风场向量（箭头/颜色） =====================
// 绘制风场RT的可视化结果（简化版：用颜色表示向量方向，亮度表示风速）
void renderWindField(GLuint windRT)
{
    // 简单的可视化Shader（顶点+片段）
    const char* vertSource = R"(
        #version 430 core
        layout(location = 0) in vec2 pos;
        layout(location = 1) in vec2 texCoord;
        out vec2 vTexCoord;
        void main() {
            gl_Position = vec4(pos, 0.0, 1.0);
            vTexCoord = texCoord;
        }
    )";

    const char* fragSource = R"(
        #version 430 core
        in vec2 vTexCoord;
        uniform sampler2D windRT;
        out vec4 fragColor;

        // 将向量转换为HSV颜色（H=方向，V=风速归一化）
        // vec3 hsv2rgb(vec3 hsv) {
        //     float h = hsv.x / 360.0;
        //     float s = hsv.y;
        //     float v = hsv.z;

        //     vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
        //     vec3 p = abs(fract(h + K.xyz) * 6.0 - K.www);
        //     return v * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), s);
        // }

        void main() {
            // 读取风向向量
            vec2 windVec = texture(windRT, vTexCoord).rg;
            // float speed = length(windVec);

            // // 向量为0则显示黑色
            // if (speed < 0.001) {
            //     fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            //     return;
            // }

            // // 计算方向（角度）：0°=右，90°=上，180°=左，270°=下
            // float angle = atan(windVec.y, windVec.x) * 180.0 / 3.1415926535;
            // if (angle < 0.0) angle += 360.0;

            // // HSV转RGB：H=角度，S=1，V=风速归一化（最大风速设为10）
            // vec3 rgb = hsv2rgb(vec3(angle, 1.0, min(speed / 10.0, 1.0)));
            fragColor = vec4(windVec, 0.0, 1.0);
        }
    )";

    // 编译可视化Shader（简化：直接创建，不封装）
    GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertShader, 1, &vertSource, NULL);
    glCompileShader(vertShader);

    GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragShader, 1, &fragSource, NULL);
    glCompileShader(fragShader);

    GLuint visProgram = glCreateProgram();
    glAttachShader(visProgram, vertShader);
    glAttachShader(visProgram, fragShader);
    glLinkProgram(visProgram);

    // 全屏四边形VAO/VBO
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    // 顶点数据：pos(xy) + texCoord(xy)
    float vertices[] = {-1.0f, -1.0f, 0.0f, 0.0f, 1.0f,  -1.0f, 1.0f, 0.0f,
                        1.0f,  1.0f,  1.0f, 1.0f, -1.0f, 1.0f,  0.0f, 1.0f};
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    // 绘制全屏四边形
    glUseProgram(visProgram);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, windRT);
    glUniform1i(glGetUniformLocation(visProgram, "windRT"), 0);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    // 释放临时资源
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(visProgram);
    glDeleteShader(vertShader);
    glDeleteShader(fragShader);
}

// ===================== 主函数 =====================
int main()
{
    // 初始化GLFW
    if (!glfwInit())
    {
        std::cerr << "GLFW初始化失败" << std::endl;
        return -1;
    }

    // 创建窗口
    GLFWwindow* window = glfwCreateWindow(RT_WIDTH, RT_HEIGHT, "Wind Field", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();

    // 初始化资源
    initWindRT();
    initUBO();
    initComputeShader();

    // ===================== 初始化风场形状 =====================
    windParams.shapeCount = 3;
    windParams.rtWidth = RT_WIDTH;
    windParams.rtHeight = RT_HEIGHT;

    // 形状1：圆形风场（中心(300,400)，半径100，风向向右上，风速5，衰减0.5）
    windParams.shapes[0].type = SHAPE_CIRCLE;
    windParams.shapes[0].pos = glm::vec2(200.0f, 300.0f);
    windParams.shapes[0].size = glm::vec2(100.0f, 0.0f); // 圆形：size.x=半径
    windParams.shapes[0].rotation = 0.0f;
    windParams.shapes[0].angleRange = 0.0f;
    windParams.shapes[0].windDir = glm::normalize(glm::vec2(0.5f, 1.0f)); // 右上
    windParams.shapes[0].windSpeed = 0.5f;

    // 形状2：矩形风场（中心(500,300)，尺寸200x100，旋转45°，风向向左，风速8，衰减0.8）
    windParams.shapes[1].type = SHAPE_RECT;
    windParams.shapes[1].pos = glm::vec2(300.0f, 200.0f);
    windParams.shapes[1].size = glm::vec2(200.0f, 100.0f); // 宽200，高100
    windParams.shapes[1].rotation = 45.0f;                 // 旋转45°
    windParams.shapes[1].angleRange = 0.0f;
    windParams.shapes[1].windDir = glm::normalize(glm::vec2(1.0f, 0.5f)); // 向左
    windParams.shapes[1].windSpeed = .8f;

    // 形状3：扇形风场（中心(400,500)，半径150，起始角度30°，范围120°，风向向下，风速6，衰减0.3）
    windParams.shapes[2].type = SHAPE_SECTOR;
    windParams.shapes[2].pos = glm::vec2(400.0f, 500.0f);
    windParams.shapes[2].size = glm::vec2(150.0f, 0.0f);                  // 扇形：size.x=半径
    windParams.shapes[2].rotation = 30.0f;                                // 起始角度30°
    windParams.shapes[2].angleRange = 120.0f;                             // 角度范围120°（终止角度150°）
    windParams.shapes[2].windDir = glm::normalize(glm::vec2(0.0f, 0.3f)); // 向下
    windParams.shapes[2].windSpeed = .6f;

    // 更新UBO数据
    glBindBuffer(GL_UNIFORM_BUFFER, uboParams);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(WindFieldParams), &windParams);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

    // ===================== 主循环 =====================
    while (!glfwWindowShouldClose(window))
    {
        // 步骤1：调度Compute Shader计算风场向量
        glUseProgram(computeProgram);
        glBindImageTexture(1, windRT, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
        // 启动计算：(宽+15)/16 × (高+15)/16 个工作组
        glDispatchCompute((RT_WIDTH + 15) / 16, (RT_HEIGHT + 15) / 16, 1);
        // 等待计算完成（确保RT写入完成）
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        // 步骤2：清空屏幕，渲染风场可视化结果
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        renderWindField(windRT);

        // 交换缓冲区，处理事件
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // ===================== 释放资源 =====================
    glDeleteProgram(computeProgram);
    glDeleteTextures(1, &windRT);
    glDeleteBuffers(1, &uboParams);
    glfwTerminate();

    return 0;
}
