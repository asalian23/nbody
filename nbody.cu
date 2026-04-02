#include <iostream>
#include <vector>
#include <cmath>
#include <deque>
#include <cuda_runtime.h>
#include <SFML/Graphics.hpp>
#include <random>

//10k
//Barnes-Hut
//3D
//Collisions
//compile gui engine into a dll and use python UI side, goal is to seperating the gui side and the calculation side between python and cpp
//Data vizualition with python using CUDA reduction kernel
//use Pybind11, library to expose cpp functions to python, connecting c++ with python
//import nbody_engine

//Set
__constant__ double G_device = 6.67430e-11;

const double G_host = 6.67430e-11;
const double Pi = 3.14159265358979323846;

//Alterable
const int N = 10000; //number of bodies

__constant__ double dt = 10; //time progressed per frame in seconds
__constant__ double epsilon_device = 6.96e8; //sun radius in meters
__constant__ double theta_device = 0.5;

const double epsilon_host = 6.96e8;
const double theta_host = 0.5;
const int maxDepth = 21; //ensures we don't have a stack overflow in the gpu
const int maxNodes = N*8; //worst case node usage, each body insertion creates at most 4 bodies, and then there's some padding.
const double maxRadius = 3 * 1.496e11; //3 AU
const double Scale = 600.0/maxRadius; //screen width some margins divided by the max radius
float centerPx = 1280.0f;
float centerPy = 720.0f;

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

struct quadNode {
    Vec3 centerOfMass;
    Vec3 centerPos;
    double totalMass, size; 
    int children[4], bodyIdx;
};

void resetNode(quadNode& node) {
    node.centerOfMass = Vec3();
    node.centerPos = Vec3();
    node.totalMass = 0;
    node.size = 0;
    node.bodyIdx = -1;
    for (int i=0; i<4; i++) node.children[i] = -1;
}

quadNode getRootNode(const std::vector<body>& bodies) {
    double xMin = bodies[0].pos.x;
    double xMax = bodies[0].pos.x;
    double yMin = bodies[0].pos.y;
    double yMax = bodies[0].pos.y;

    for (int i=0; i<bodies.size(); i++) {
        xMin = std::min(xMin, bodies[i].pos.x);
        xMax = std::max(xMax, bodies[i].pos.x);
        yMin = std::min(yMin, bodies[i].pos.y);
        yMax = std::max(yMax, bodies[i].pos.y);
    }

    quadNode root;
    resetNode(root);
    root.size = std::max(xMax-xMin, yMax-yMin) * 1.01; //Multiplying by 1.01 handles rare edge cases
    root.centerPos = Vec3((xMin + xMax)/2.0, (yMin + yMax)/2.0, 0);
    
    return root;
}

int getQuadrant(const body& b, const quadNode& node) {
    return (b.pos.x >= node.centerPos.x) | ((b.pos.y >= node.centerPos.y) << 1); //using bitwise to avoid branching
}

void calcChildCenterPos (const quadNode& parent, const int& quadrant, Vec3& childCenterPos) {
    int xBit = quadrant & 1; //extract bits, the & operator only keeps the int n bits from the left
    int yBit = (quadrant >> 1) & 1;
    double shift = parent.size /4.0; //shift is 1/4 of parent node width

    //2*bit - 1, will turn the 0 or 1 into -1 or 1 nicely to apply shift
    childCenterPos.x = parent.centerPos.x + (2 * xBit - 1) * shift;
    childCenterPos.y = parent.centerPos.y + (2 * yBit - 1) * shift;
    childCenterPos.z = parent.centerPos.z;
}

void insertBody(std::vector<body>& bodies, std::vector<quadNode>& nodes, int& nodeCount, int bodyIdx, int nodeIdx, int currDepth) { //This function must be seperate as it recurses into itself
    const body& currBody = bodies[bodyIdx];
    quadNode& currNode = nodes[nodeIdx];

    //First case, empty node
    if (currNode.bodyIdx == -1 && currNode.children[0] == -1) {
        currNode.bodyIdx = bodyIdx;
        currNode.totalMass = currBody.mass;
        currNode.centerOfMass = currBody.pos;
        return;
    }

    //Second case, leaf node + depth check
    if (currNode.bodyIdx != -1) {
        if (currDepth > maxDepth) return; //If the masses are stored so close together we can't seperate them after 21 splits (430 km apart), drop the body. Yes its technically inaccurate but the error is minor. Also, we can fix this eventually with linked lists.
        for (int i=0; i<4; i++) {
            quadNode& currChildNode = nodes[nodeCount];
            currNode.children[i] = nodeCount; //idx of child node is next avaliable nodeIdx
            resetNode(currChildNode); //init child node
            currChildNode.size = currNode.size / 2.0; //sets child size
            calcChildCenterPos(currNode, i, currChildNode.centerPos); //sets child center
            nodeCount++;
        }
        //the previously leaf node, now parent node, still needs its bodyIdx reset. Note that it will have an incorrect mass and com value, but those will be set properly through the bottom up pass later
        int prevSetBodyIdx = currNode.bodyIdx;
        currNode.bodyIdx = -1;
        insertBody(bodies, nodes, nodeCount, prevSetBodyIdx, nodeIdx, currDepth+1); //Note that we're inserting into nodeIdx, or the parent node still, and it will recurse into the current child itself
        insertBody(bodies, nodes, nodeCount, bodyIdx, nodeIdx, currDepth+1); //Now we insert the new body
        return; //Stops this run, otherwise the new internal node will be subjected to case 3.
    }

    //Third Case, internal node
    int quadrant = getQuadrant(currBody, currNode);
    insertBody(bodies, nodes, nodeCount, bodyIdx, currNode.children[quadrant], currDepth+1);
}

void fillTree(std::vector<body>& bodies, std::vector<quadNode>& nodes, int& nodeCount) { //loops through bodies and inserts each one
    nodeCount = 1; //the root node, nodeCount represents both the number of node's allocated and the next avalaible nodeIdx
    nodes[0] = getRootNode(bodies);

    for (int bodyIdx=0; bodyIdx<N; bodyIdx++) {
        insertBody(bodies, nodes, nodeCount, bodyIdx, 0, 0);
    }
}

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
            double r = (tile[i].pos - locPos).magnitude(); //distance between bodies
            double AccelMag = (G_device*tile[i].mass)/(r*r + epsilon_device * epsilon_device); //I added a smoothing variable epsilon here, preventing the forces from exploding if the object gets to close to the central mass
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

//same two trail functions from CPU code
void updateTrail(std::deque<Vec3>& trail, const body& b, int maxLength) {
    trail.push_front(b.pos);
    if (trail.size() > maxLength) {
        trail.pop_back();
    }
}

void drawTrail(sf::RenderWindow& window, std::deque<Vec3>& trail, sf::Color color, double Scale) {
    for (int i=0; i<trail.size(); i++) {
        int opacity = .125 * 255 * (1.0 - (float)i / trail.size()); //1-i/size decreases the opacity along the trail
        sf::CircleShape dot(3.f);
        sf::Color fadingColor(color.r, color.g, color.b, opacity); //keeping the rgb and just adjusting opacity
        dot.setFillColor(fadingColor); //couldnt just input rgb and opacity here as it only takes a color input
        float dotx = centerPx+ trail[i].x * Scale; //using floats as its just mapping to pixels, not much impact in this stage though
        float doty = centerPx + trail[i].y * Scale;
        dot.setPosition({dotx - 3.f, doty - 3.f}); //offsetting by the radius as the function sets the top left corner
        window.draw(dot);
    }
}

std::vector<body> initRandBodies(int N) {
    std::vector<body> bodies;

    std::mt19937 rng(23); //set seed for replicable results
    std::uniform_real_distribution<double> angleRange(0, 2 * Pi);
    std::uniform_real_distribution<double> radiusRange(0, maxRadius);

    body centralBody; //pos and vel is 0 by default
    centralBody.mass = 1e38; //very big black hole
    bodies.push_back(centralBody);

    for (int i=0; i<N; i++) {
        body temp;
        temp.mass = 1.989e30; //set mass for all bodies to the same mass, the sun, for now because otherwise initial velocity would be difficult to do
        double ang = angleRange(rng); //randomizing through polar and then converting to cartesian
        double r = radiusRange(rng);
        temp.pos = Vec3(r*cos(ang), r*sin(ang), 0); //initialize a random position
        double vel = sqrt(G_host * centralBody.mass/r);
        temp.vel = Vec3(vel * (-temp.pos.y/r), vel * (temp.pos.x/r), 0); //Finds the unit vector perpendicular to the radius and scales it to velocity
        bodies.push_back(temp);
    }
    return bodies;
}

int main() {
    //generate N random bodies
    std::vector<body> bodies = initRandBodies(N);
    //GPU prep
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
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return 0;
}