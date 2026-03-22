#include <iostream>
#include <vector>
#include <cmath>
#include <deque>
#include <cuda_runtime.h>
#include <SFML/Graphics.hpp>

__constant__ double G = 6.67430e-11;
__constant__ double dt = 3600; //time per loop in seconds
double scale = 300.0/1.496e11; //only used for drawing so just a double, this is px/AU, so earth would be 300 px away from the sun

struct Vec3 {
    double x, y, z;

    __host__ __device__ Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    __host__ __device__ Vec3 operator*(double s)      const { return {x*s,   y*s,   z*s};   }

    __host__ __device__ double magnitude() const { return sqrt(x*x + y*y + z*z); }
    __host__ __device__ double dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }

    __host__ __device__ Vec3 uVec() const {
        double mag = magnitude();
        return {x/mag, y/mag, z/mag};
    }
};

struct body {
    Vec3 pos;
    Vec3 vel;
    Vec3 acc;
    double mass;
};

__global__ void calcAccel(body* bodies, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //global index
    if (idx>=N) return; //bounds check
    Vec3 locAcc(0, 0, 0); //local var to accumalate the acceleration, this avoids reading from slower VRAM each time

    for (int i=0; i<N; i++) {
        if (idx == i) continue; //skips itself
        double r = (bodies[i].pos - bodies[idx].pos).magnitude(); //distance between bodies
        double AccelMag = (G*bodies[i].mass)/(r*r);
        Vec3 dir((bodies[i].pos - bodies[idx].pos).uVec()); //direction from body idx to body i
        locAcc = locAcc + dir*AccelMag; //makes accel into a vector and then accumalates it
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

//same two trail functions from CPU code
void updateTrail(std::deque<Vec3>& trail, const body& b, int maxLength) {
    trail.push_front(b.pos);
    if (trail.size() > maxLength) {
        trail.pop_back();
    }
}

void drawTrail(sf::RenderWindow& window, std::deque<Vec3>& trail, sf::Color color, double scale) {
    for (int i=0; i<trail.size(); i++) {
        int opacity = .125 * 255 * (1.0 - (float)i / trail.size()); //1-i/size decreases the opacity along the trail
        sf::CircleShape dot(3.f);
        sf::Color fadingColor(color.r, color.g, color.b, opacity); //keeping the rgb and just adjusting opacity
        dot.setFillColor(fadingColor); //couldnt just input rgb and opacity here as it only takes a color input
        float dotx = 400.f + trail[i].x * scale; //using floats as its just mapping to pixels, not much impact in this stage though
        float doty = 400.f + trail[i].y * scale;
        dot.setPosition({dotx - 3.f, doty - 3.f}); //offsetting by the radius as the function sets the top left corner
        window.draw(dot);
    }
}

int main() {
    //Set up bodes vector
    std::vector<body> bodies;

    body earth;
    earth.mass = 5.972e24;
    earth.pos  = Vec3(1.496e11, 0, 0); // 1 AU (the average distance of the earth from the sun) from origin
    earth.vel  = Vec3(0, 29783, 0);    // orbital velocity in m/s

    body sun;
    sun.mass = 1.989e30;
    sun.pos  = Vec3(0, 0, 0); 
    sun.vel  = Vec3(0, 0, 0);

    body jupiter;
    jupiter.mass = 1.898e27;
    jupiter.pos = Vec3(7.785e11, 0, 0);
    jupiter.vel = Vec3(0, 13070, 0);

    bodies.push_back(sun);
    bodies.push_back(earth);
    bodies.push_back(jupiter);

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
    std::deque<Vec3> sunT;
    std::deque<Vec3> earthT;

    sf::RenderWindow window(sf::VideoMode({800, 800}), "nbody sim");
    window.setFramerateLimit(240);

    while (window.isOpen()) {
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {   
                window.close();
            }
        }

        leapfrog(d_bodies, N, numBlocks, threadsPerBlock); //increment position each frame
        cudaDeviceSynchronize(); //finish updating bodies positions before copying back to CPU
        cudaMemcpy(bodies.data(), d_bodies, N * sizeof(body), cudaMemcpyDeviceToHost); //copy data back to CPU for drawing
        window.clear(); //clear the old frame
        
        //draw sun
        sf::CircleShape sunC(50.f);
        sunC.setFillColor(sf::Color::Yellow);
        float sx = 400.f + bodies[0].pos.x * scale;
        float sy = 400.f + bodies[0].pos.y * scale;
        sunC.setPosition({sx - 50.f, sy - 50.f}); //as it outputs at the top left, we offset so the center is at the desired point
        window.draw(sunC);

        //draw earth
        sf::CircleShape earthC(20.f);
        earthC.setFillColor(sf::Color::Blue);
        float ex = 400.f + bodies[1].pos.x * scale;
        float ey = 400.f + bodies[1].pos.y * scale;
        earthC.setPosition({ex - 20.f, ey - 20.f});
        window.draw(earthC);

        //draw and update trails
        updateTrail(earthT, bodies[1], 1250); //The variables "earth" and "sun" don't change after being pushed back so we use the versions being updated in bodies.
        updateTrail(sunT, bodies[0], 1250);
        drawTrail(window, earthT, sf::Color::Blue, scale);
        drawTrail(window, sunT, sf::Color::Yellow, scale);

        window.display();
    }

    cudaFree(d_bodies);
    return 0;
}