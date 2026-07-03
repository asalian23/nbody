#include <iostream>
#include <vector>
#include <cmath>
#include <deque>
#include <cuda_runtime.h>
#include <random>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

//10k - Done
//Barnes-Hut - Done
//3D - Done
//Collisions
//compile gui engine into a dll and use python UI side, goal is to seperating the gui side and the calculation side between python and cpp
//Data vizualition with python using CUDA reduction kernel
//use Pybind11, library to expose cpp functions to python, connecting c++ with python
//import nbody_engine
//Next project is matrix calculator

//Everything is an astronomical units or otherwise larger units as floats can only handle a max of about 3.4028235 × 10^38

//Set
__constant__ float G_device = 3.964e-14f; //G is not 6.67430e-11 as we're not using SI units

const float G_host = 3.964e-14f; // AU³ / (M_sun * s²)
const float Pi = 3.14159265358979323846f;

//Alterable
const int N = 10000; //number of bodies
//Do 2pir / v for the bodies to make sure the time step is good enough compared to the central mass
__constant__ float dt = 10; //time progressed per frame in seconds
__constant__ float epsilon_device = 4.65e-3f; //sun radius in AU
__constant__ float theta_device = 0.5f;

const float maxRadius = 6.0f;
const float Scale = 600.0f/maxRadius; //screen width some margins divided by the max radius
float centerPx = 640.0f; //note that this should be adjusted by screen on the github
float centerPy = 400.0f;

struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    __host__ __device__ Vec3 operator*(float s)      const { return {x*s,   y*s,   z*s};   }

    __host__ __device__ float magnitude() const { return sqrt(x*x + y*y + z*z); }
    __host__ __device__ float dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }

    __host__ __device__ Vec3 uVec() const {
        float mag = magnitude();
        return {x/mag, y/mag, z/mag};
    }
};

struct body {
    Vec3 pos;
    Vec3 vel;
    Vec3 acc;
    float mass;
};

__global__ void calcAccel(body* bodies, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //global index
    __shared__ body tile[256]; //making shared memory space for one block
    if (idx>=N) return; //bounds check
    Vec3 locAcc(0, 0, 0); //local var to accumalate the acceleration, this avoids reading from slower VRAM each time
    Vec3 locPos = bodies[idx].pos;  //saves one read from global
    
    for (int tileStartIdx=0; tileStartIdx<N; tileStartIdx+=256) { //loop through tiles
        int globalLoadIdx = tileStartIdx + threadIdx.x; //This is so each thread knows which body from global to load into SM
        if (globalLoadIdx < N) { //If we're in bounds
            tile[threadIdx.x] = bodies[globalLoadIdx]; //load from full bodies array in global to SM
        }
        else {
            tile[threadIdx.x] = body(); //if out of bounds just fill with an empty body, we can't return or break because then syncthreads() wont work.
        }
        
        __syncthreads();

        for (int i=0; i<256 && tileStartIdx + i < N; i++) { //now instead of looping through bodies we loop through the tile in SM, also making sure we skip the out of bounds bodies
            if (idx == tileStartIdx + i) continue; //skips itself
            Vec3 rVec = (tile[i].pos - locPos); //vector between bodies
            float rSmoothedSquared = rVec.dot(rVec) + epsilon_device * epsilon_device; //So because we're using floats, bodies can be overlapping each other makes rVec zero, meaning we can't run magnitude due to NaN poisoning
            float rSmoothedInverse = rsqrt(rSmoothedSquared);
            //float AccelMag = (G_device*tile[i].mass)/(rSmoothedSquared); //I added a smoothing variable epsilon here, preventing the forces from exploding if the object gets to close to the central mass
           // Vec3 dir(rVec * rSmoothedInverse); //direction from body idx to body i, same problem here with uVec, NaN poisoning as it also uses magnitude function in uVec. Replaced with new logic to get direction.
            locAcc = locAcc + (rVec * rSmoothedInverse) * G_device * tile[i].mass * rSmoothedInverse * rSmoothedInverse; //makes accel into a Vec3 and then accumalates it
        }
    }
    bodies[idx].acc = locAcc;
}

__global__ void leapfrogPartOne(body* bodies, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>=N) return;
    
    Vec3 locVel = bodies[idx].vel + bodies[idx].acc * 0.5 * dt; //Pulls to register and applies kick 1, saving one global memory read
    bodies[idx].pos = bodies[idx].pos + locVel*dt; //Applies drift and writes to global(VRAM)
    bodies[idx].vel = locVel; //writes vel
}

__global__ void leapfrogPartTwo(body* bodies, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>=N) return;

    bodies[idx].vel = bodies[idx].vel + bodies[idx].acc * 0.5 * dt; //Applies kick 2
}

void leapfrog(body* d_bodies, int numBlocks, int threadsPerBlock) {
    //Kick 1 + Drift and bring updated bodies back to cpu
        leapfrogPartOne<<<numBlocks, threadsPerBlock>>>(d_bodies, N);
        calcAccel<<<numBlocks, threadsPerBlock>>>(d_bodies, N);
        leapfrogPartTwo<<<numBlocks, threadsPerBlock>>>(d_bodies, N);
}

std::vector<body> initRandBodies(int N) {
    std::vector<body> bodies;

    std::mt19937 rng(23); //set seed for replicable results
    std::uniform_real_distribution<float> angleRange(0, 2 * Pi);
    std::uniform_real_distribution<float> radiusRange(0.5f, maxRadius); //Everything spawns at half an AU away from the central mass at a minimum
    std::uniform_real_distribution<float> zRange(-3.0f, 3.0f); //this is set for a flatish galaxy type simulation, if I want to do a sphere I need to add phi

    body centralBody; //pos and vel is 0 by default
    centralBody.mass = 1e8f; //very big black hole
    bodies.push_back(centralBody);

    for (int i=0; i<N-1; i++) { //our memcpy and malloc function copy exactly N, and with the central body we have N+1 bodies, so we just randomize N-1 bodies.
        body temp;
        temp.mass = 1.0f; //set mass for all bodies to the same mass, the sun, for now because otherwise initial velocity would be difficult to do
        float ang = angleRange(rng); //randomizing through polar and then converting to cartesian
        float r = radiusRange(rng);
        float z = zRange(rng);
        temp.pos = Vec3(r*cos(ang), r*sin(ang), z); //initialize a random position
        float vel = sqrt(G_host * centralBody.mass/r);
        temp.vel = Vec3(vel * (-temp.pos.y/r), vel * (temp.pos.x/r), 0); //Finds the unit vector perpendicular to the radius and scales it to velocity
        bodies.push_back(temp);
    }
    return bodies;
}

GLuint compileShaderProgram() {
    const char* vertSrc = R"(
        #version 460 core
        layout(location = 0) in vec2 aPos;
        layout(location = 1) in vec3 aColor;
        out vec3 vColor;
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
            gl_PointSize = 1.0;
            vColor = aColor;
        }
    )";

    const char* fragSrc = R"(
        #version 460 core
        in vec3 vColor;
        out vec4 FragColor;
        void main() {
            FragColor = vec4(vColor, 1.0);
        }
    )";

    GLuint vert = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vert, 1, &vertSrc, NULL);
    glCompileShader(vert);

    GLuint frag = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(frag, 1, &fragSrc, NULL);
    glCompileShader(frag);

    GLuint program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);

    glDeleteShader(vert);
    glDeleteShader(frag);
    return program;
}

//writes data from CUDA to vbo but also handles camera and screen pos math
__global__ void writeToVBO(body* bodies, float* vbo, float cosX, float sinX, float cosY, float sinY, float camDist, float scale, float cx, float cy, float halfW, float halfH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //global idx + bounds check
    if (idx >= N) return;
    Vec3 pos = bodies[idx].pos; //Pulling pos to local var to save global memory reads

    //roation about the x-axis
    float y1 = pos.y * cosX - pos.z * sinX;
    float z1 = pos.y * sinX + pos.z * cosX;

    //rotation abt y-axis
    float x2 = pos.x * cosY + z1 * sinY;
    float z2 = -pos.x * sinY + z1 * cosY;

    float perspective = camDist/(camDist+z2); //Similar triangle side ratio used to project the body along its ray towards the camera onto z=0

    float screenPX = cx + x2 * scale * perspective; //x2 is in AU, scale converts it to px, then perpspective applies the depth. This is all relative to the center though, so we add cx, the center pixel
    float screenPY = cy + y1 * scale * perspective; //same logic

    int vboIdx = idx*5; //5 floats per body, px, py, r,g,b, Note - this is refering to the first term in the 5 floats, the px

    vbo[vboIdx] = (screenPX / halfW) - 1.0f; //OpenGl doesnt take in pixels, instead running from -1 (left edge) to 1 (right edge), so we divide the pixel by half the screen width, getting it on a scale from 0 to 2, and subtract 1 to make it range from -1, 1
    vbo[vboIdx + 1] =  1.0f - (screenPY / halfH); //adding one makes this refer to the second term, py. Also for px, y increases from down to up but its up to down in openGL, so we invert it, otherwise same logic

    if (idx == 0) { //setting the black hole to yellow
        vbo[vboIdx + 2] = 1.0f;
        vbo[vboIdx + 3] = 0.78f;
        vbo[vboIdx + 4] = 0.2f;
    } else {
        vbo[vboIdx + 2] = 1.0f; //everything else to white
        vbo[vboIdx + 3] = 1.0f;
        vbo[vboIdx + 4] = 1.0f;
    }

}

int main() {
    //generate N random bodies
    std::vector<body> bodies = initRandBodies(N);
    //Init nodes vector
    //std::vector<octNode> nodes(maxNodes); No longer needed as all on GPU
    //GPU prep
    int threadsPerBlock = 256;
    int numBlocks = (N+threadsPerBlock-1)/threadsPerBlock;
    //Prep for main logic
    //Bodies prep and Morton Code prep
    body* d_bodies;
    cudaMalloc(&d_bodies, N * sizeof(body)); //Reserve VRAM for bodies
    cudaMemcpy(d_bodies, bodies.data(), N*sizeof(body), cudaMemcpyHostToDevice); //Copies vector to GPU

    //Nodes Prep
    
    
    calcAccel<<<numBlocks, threadsPerBlock>>>(d_bodies, N); //assign initial acceleration to bodies
    
    //Creating window and loading OpenGL Functions
    glfwInit(); //starts library
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); //Setting version to 4.6 for GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6); 
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); //Selects Core profile
    GLFWwindow* window = glfwCreateWindow(1280, 800, "nbody sim", NULL, NULL); //creating the window, the first NULL is for passing a monitor and the second NULL is for sharing openGl over mutliple windows
    glfwMakeContextCurrent(window); //sets the drawing window
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress); //loads gl__ functions from GPU driver so they can be used normally in code
    glEnable(GL_PROGRAM_POINT_SIZE); //allows changing point sizes to more than 1px

    //VAO and VBO setup
    GLuint vao, vbo; //creates unsigned ints that will refer to the vertex array object and vertex buffer object
    glGenVertexArrays(1, &vao); //creating vao and setting Gluint vao equal to its id, 1 is just the number of vaos we want, we would use an array if we wanted multiple rather than an int
    glBindVertexArray(vao); //makes the vao active, meaning configurations are now recorded in this vao

    glGenBuffers(1, &vbo); //same thing but for vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo); //makes the buffer active, the GL_ARRAY_BUFFER just tells OpenGL that the buffer is holding vertex data
    glBufferData(GL_ARRAY_BUFFER, N*5*sizeof(float), nullptr, GL_DYNAMIC_DRAW); //we reserve GPU memory for a buffer of vertex data that has 5 floats per body, is filled with null atm, and GL_DYNAMIC_DRAW tells the GPU driver the buffer will be frequently updated so it places it in a more optimal location. The 5*float size is the stride, how much to jump for the next vector

    //configuring how OpenGL interprets the floats in the buffer, this is for position
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0); //configuring location 0, read 2 floats from the vbo starting at byte 0, and GL_FALSE means dont normalize values, or dont turn rgb vals into floats
    glEnableVertexAttribArray(0);

    //Now setting color
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(2*sizeof(float))); //same thing but the first byte is set 2 floats worth (8bytes) to the right
    glEnableVertexAttribArray(1);

    //compiling shader program
    GLuint shaderProgram = compileShaderProgram(); //compiles vertex and fragment shader, linking them, and stores program id

    //Link vbo with CUDA so kernels can write to it
    cudaGraphicsResource* cudaVBO; //pointer so CUDA can track buffer
    cudaGraphicsGLRegisterBuffer(&cudaVBO, vbo, cudaGraphicsMapFlagsWriteDiscard); //Allows CUDA to write into vbo, flag "cudaGraphics..." means everything will be overwritten per write

    //camera constants
    float camDist = 9.0f; //camera's distance away from origin along the z-axis
    float tiltX = 0.6f, tiltY = 0.25f; //xy axis tilts so view isnt flat
    float cosX = cos(tiltX), sinX = sin(tiltX), cosY = cos(tiltY), sinY = sin(tiltY);

    //benchmarking inits
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    int frameCount=0;
    float frameTimes[100];
    
    //we dont need cudaDeviceSynchronize() here as the leapfrog CPU function is only made of kernels

    while (!glfwWindowShouldClose(window)) { //like SFML, keep going if no close event occurs
        glfwPollEvents(); //keep processing OS queues

            cudaEventRecord(t0); //record starting time

            leapfrog(d_bodies, numBlocks, threadsPerBlock); //increment pos and vel, and updates acceleration
            
            cudaEventRecord(t1);
            cudaEventSynchronize(t1); //stops CPU until GPU actually finishes the action
            float ms;
            cudaEventElapsedTime(&ms, t0, t1);

            if (frameCount >= 20 && frameCount < 120) {
                frameTimes[frameCount-20] = ms;
            }
            if (frameCount == 120) {
                float total = 0;
                for (int i=0; i<100; i++) total += frameTimes[i];
                printf("Average: %.2f ms/frame\n", total / 100.0f);
            }
            frameCount++;

            float* d_vboPtr; //Will hold GPU memory address of the VBO
            size_t bufferSize; //will recieve the size of the buffer in bytes, size_t is a unsigned integer type specifically for byte counts, and its used because CUDA uses it
            cudaGraphicsMapResources(1, &cudaVBO, 0); //Gives CUDA control over buffer, remember cudaVBO is the OpenGL type ptr to the buffer, the 0 basically means wait for all openGL actions to complete before executing this
            //Note - The above function is just internal semantics, telling whether CUDA or OpenGL is about to read and write, it does not move or change any data
            cudaGraphicsResourceGetMappedPointer((void**)&d_vboPtr, &bufferSize, cudaVBO); //writes the ptr for the buffer into d_vboPtr and the size into bufferSize, passing a ptr by reference because we want to change where the ptr is pointing to, not the value the variable

            writeToVBO<<<numBlocks, threadsPerBlock>>>(d_bodies, d_vboPtr, cosX, sinX, cosY, sinY, camDist, Scale, centerPx, centerPy, 640.0f, 400.0f); //actually write the position data into the buffer

            cudaGraphicsUnmapResources(1, &cudaVBO, 0); //Giving control back to OpenGL for rendering, also making sure writeToVBO is fully completed before OpenGL can read the buffer, thats why cudaDeviceSync is not needed here

            //Drawing
            glClear(GL_COLOR_BUFFER_BIT); //clear buffer
            glUseProgram(shaderProgram); //activates compiler vertex and fragment shaders
            glBindVertexArray(vao); //restores all configurations so OpenGL can read the buffer correctly
            glDrawArrays(GL_POINTS, 0, N); //run rendering pipeline from 0 through N-1 verticies from VAO. For each vertex in vao, 5 floats from VBO are read, shaders are run, and pxs are written to back buffer.
            glfwSwapBuffers(window); //switches back buffer to visible one.

        }


    cudaGraphicsUnregisterResource(cudaVBO); //breaks CUDA - OpenGL link
    glDeleteBuffers(1, &vbo); //deletes the buffers
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(shaderProgram);
    glfwDestroyWindow(window); //deletes the window and shuts off glfw
    glfwTerminate();
    cudaFree(d_bodies); //clear up GPU memory
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return 0;
}