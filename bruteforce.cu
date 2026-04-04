#include <iostream>
#include <vector>
#include <cmath>
#include <deque>
#include <cuda_runtime.h>
#include <SFML/Graphics.hpp>
#include <random>
//cd "C:\Users\aryan\OneDrive\Documents\nBody"

//Set
__constant__ float G_device = 6.67430e-11;
const float G_host = 6.67430e-11;
const float Pi = 3.14159265358979323846;

//Alterable
__constant__ float dt = 10; //time progressed per frame in seconds
__constant__ float epsilon_device = 6.96e8; // softing
const float maxRadius = 3 * 1.496e11; //3 AU
const float Scale = 600.0/maxRadius; //screen width some margins divided by the max radius
float centerPx = 1280.0f;
float centerPy = 720.0f;

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
            float r = (tile[i].pos - locPos).magnitude(); //distance between bodies
            float AccelMag = (G_device*tile[i].mass)/(r*r);
            Vec3 dir((tile[i].pos - locPos).uVec()); //direction from body idx to body i
            locAcc = locAcc + dir*AccelMag; //makes accel into a vector and then accumalates it
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

void leapfrog(body* d_bodies, int N, int numBlocks, int threadsPerBlock) {
    leapfrogPartOne<<<numBlocks, threadsPerBlock>>>(d_bodies, N); //Applies kick 1 and drift
    calcAccel<<<numBlocks, threadsPerBlock>>>(d_bodies, N); //Calculates next acceleration
    leapfrogPartTwo<<<numBlocks, threadsPerBlock>>>(d_bodies, N); //Applies kick 2
}

std::vector<body> initRandBodies(int N) {
    std::vector<body> bodies;

    std::mt19937 rng(23); //set seed for replicable results
    std::uniform_real_distribution<float> angleRange(0, 2 * Pi);
    std::uniform_real_distribution<float> radiusRange(1e9f, maxRadius); //added minimum spawn radius

    body centralBody; //pos and vel is 0 by default
    centralBody.mass = 1e38; //very big black hole
    bodies.push_back(centralBody);

    for (int i=0; i<N; i++) {
        body temp;
        temp.mass = 1.989e30; //set mass for all bodies to the same mass, earth, for now because otherwise initial velocity would be difficult to do
        float ang = angleRange(rng); //randomizing through polar and then converting to cartesian
        float r = radiusRange(rng);
        temp.pos = Vec3(r*cos(ang), r*sin(ang), 0); //initialize a random position
        float vel = sqrt(G_host * centralBody.mass/r);
        temp.vel = Vec3(vel * (-temp.pos.y/r), vel * (temp.pos.x/r), 0); //Finds the unit vector perpendicular to the radius and scales it to velocity
        bodies.push_back(temp);
    }
    return bodies;
}

int main() {
    //generate N random bodies
    std::vector<body> bodies = initRandBodies(1000000);
    //GPU prep
    int N = bodies.size();
    int threadsPerBlock = 256;
    int numBlocks = (N+threadsPerBlock-1)/threadsPerBlock;
    
    //Prep for main logic
    body* d_bodies;
    cudaMalloc(&d_bodies, N * sizeof(body)); //Reserve VRAM
    cudaMemcpy(d_bodies, bodies.data(), N*sizeof(body), cudaMemcpyHostToDevice); //Copies vector to GPU
    calcAccel<<<numBlocks, threadsPerBlock>>>(d_bodies, N); //assign initial acceleration to bodies
    //we dont need cudaDeviceSynchronize() here as the leapfrog CPU function is only made of kernels

    //trail pos storage
    //std::deque<Vec3> sunT;
    //std::deque<Vec3> earthT;

    sf::RenderWindow window(sf::VideoMode({2560, 1440}), "nbody sim");
    window.setFramerateLimit(60);

    //inits for benchmarking
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    int frameCount = 0;

    while (window.isOpen()) {
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {   
                window.close();
            }
        }
        cudaEventRecord(t0); //record starting time

        leapfrog(d_bodies, N, numBlocks, threadsPerBlock); //increment position each frame

        cudaEventRecord(t1); //record ending time
        cudaEventSynchronize(t1);
        float ms;
        cudaEventElapsedTime(&ms, t0, t1); //saves elapsed time to ms

        frameCount++;
        if (frameCount % 10 == 0) { //prints ms per frame every x frames to avoid console clog
            printf("N=%d  %.2f ms/frame\n", N, ms);
        }

        cudaDeviceSynchronize(); //finish updating bodies positions before copying back to CPU
        cudaMemcpy(bodies.data(), d_bodies, N * sizeof(body), cudaMemcpyDeviceToHost); //copy data back to CPU for drawing (cpu bottleneck), fix with OpenGL vbo later
        window.clear(); //clear the old frame
        
        sf::VertexArray points(sf::PrimitiveType::Points, N); //VectorArray requires one draw call per frame rather than bodies draw calls per frame in previous logic
        for (int i=0; i<N; i++) {
            float px = centerPx + bodies[i].pos.x * Scale;
            float py = centerPy + bodies[i].pos.y * Scale;
            points[i].position = {px, py};
            points[i].color = sf::Color::White;
        }
        window.draw(points);

        /*
        //draw and update trails
        updateTrail(earthT, bodies[1], 1250); //The variables "earth" and "sun" don't change after being pushed back so we use the versions being updated in bodies.
        updateTrail(sunT, bodies[0], 1250);
        drawTrail(window, earthT, sf::Color::Blue, Scale);
        drawTrail(window, sunT, sf::Color::Yellow, Scale); */

        window.display();
    }

    cudaFree(d_bodies);
    return 0;
}