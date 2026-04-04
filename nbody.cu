#include <iostream>
#include <vector>
#include <cmath>
#include <deque>
#include <cuda_runtime.h>
#include <SFML/Graphics.hpp>
#include <random>

//10k - Done
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
const int N = 100000; //number of bodies

__constant__ double dt = 10; //time progressed per frame in seconds
__constant__ double epsilon_device = 6.96e8; //sun radius in meters
__constant__ double theta_device = 0.5;

const double epsilon_host = 6.96e8;
const double theta_host = 0.5;
const int maxDepth = 21; //ensures we don't have a stack overflow in the gpu
const int maxNodes = N*8; //worst case node usage, each body insertion creates at most 4 bodies, and then there's some padding.
const double maxRadius = 15 * 1.496e11; //first number AU
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

//The next function also builds the tree. Note - This is a sparse tree, so watch out for nonexistent nodes when recursing
void insertBody(std::vector<body>& bodies, std::vector<quadNode>& nodes, int& nodeCount, int bodyIdx, int nodeIdx, int currDepth) { //This function must be seperate as it recurses into itself
    const body& currBody = bodies[bodyIdx];
    quadNode& currNode = nodes[nodeIdx];

    bool hasChildren = (currNode.children[0] != -1 || currNode.children[1] != -1 || currNode.children[2] != -1 || currNode.children[3] != -1);

    //First case, empty node
    if (currNode.bodyIdx == -1 && !hasChildren) { //If a node has no body in it and no children it is an empty node
        currNode.bodyIdx = bodyIdx;
        currNode.totalMass = currBody.mass;
        currNode.centerOfMass = currBody.pos;
        return;
    }

    //Second case, leaf node + depth check
    if (currNode.bodyIdx != -1) {
        if (currDepth > maxDepth) return; //If the masses are stored so close together we can't seperate them after 21 splits (430 km apart), drop the body. Yes its technically inaccurate but the error is minor. Also, we can fix this eventually with linked lists.
        
        //the previously leaf node, now parent node, still needs its bodyIdx reset. Note that it will have an incorrect mass and com value, but those will be set properly through the bottom up pass later
        int prevBodyIdx = currNode.bodyIdx;
        int prevBodyQuadrant = getQuadrant(bodies[prevBodyIdx], currNode); //prev body quadrant used to init the space for the prev body
        currNode.bodyIdx = -1;
        
        //The following logic is to readd the previous body to the appropriate child node
        quadNode& currChildNode = nodes[nodeCount]; //sets the childnodes location to next avaliable spot in nodes array
        currNode.children[prevBodyQuadrant] = nodeCount; //idx of child node is next avaliable nodeIdx
        resetNode(currChildNode); //init child node
        currChildNode.size = currNode.size / 2.0; //sets child size
        calcChildCenterPos(currNode, prevBodyQuadrant, currChildNode.centerPos); //sets child center
        nodeCount++;

        //For the following insertion we are still passing the parent's nodeIdx to let case 3 handle recursing into the child nodes, and as such we do not increment the depth
        insertBody(bodies, nodes, nodeCount, prevBodyIdx, nodeIdx, currDepth);
        //Not returning here like in the dense tree, for the sparse tree we don't know which node needs to be initialized for the new body, so we let case 3 handle that and let the parent node naturally fall into it.
    }

    //Third Case, internal node
    int quadrant = getQuadrant(currBody, currNode);

    if (currNode.children[quadrant] == -1) {
        quadNode& currChildNode = nodes[nodeCount]; //sets the childnodes location to next avaliable spot in nodes array
        currNode.children[quadrant] = nodeCount; //idx of child node is next avaliable nodeIdx
        resetNode(currChildNode); //init child node
        currChildNode.size = currNode.size / 2.0; //sets child size
        calcChildCenterPos(currNode, quadrant, currChildNode.centerPos); //sets child center
        nodeCount++;
    }
    insertBody(bodies, nodes, nodeCount, bodyIdx, currNode.children[quadrant], currDepth+1);
}

void fillTree(std::vector<body>& bodies, std::vector<quadNode>& nodes, int& nodeCount) { //loops through bodies and inserts each one
    for (int i=0; i<nodeCount; i++) resetNode(nodes[i]); //resets the old nodes

    nodeCount = 1; //the root node, nodeCount represents both the number of node's allocated and the next avalaible nodeIdx
    nodes[0] = getRootNode(bodies);

    for (int bodyIdx=0; bodyIdx<N; bodyIdx++) {
        insertBody(bodies, nodes, nodeCount, bodyIdx, 0, 0);
    }
}

void computeCOM(std::vector<quadNode>& nodes, int nodeIdx) { //nodeIdx is just read not changed so we don't pass by reference
    quadNode& currNode = nodes[nodeIdx];

    bool hasChildren = (currNode.children[0] != -1 || currNode.children[1] != -1 || currNode.children[2] != -1 || currNode.children[3] != -1);
    
    //Case 1, empty node
    if (currNode.bodyIdx == -1 && !hasChildren) {
        return;
    }

    //Case 2, leaf node
    if (currNode.bodyIdx != -1) {
        return; //we also don't do anything here as its com and total mass are set correctly already in the insertBody function
    }

    //Case 3, internal node
    double totalMass = 0.0;
    Vec3 weightedMass;
    for (int i=0; i<4; i++) { //go through each child
        if (currNode.children[i] != -1) { //will cause error trying to acces nodes[-1] if the child doesn't exist because sparse trees don't initialize empty nodes
            computeCOM(nodes, currNode.children[i]); //recurse down to compute com for each child, and then those recurse and so on, making this a bottom up search
            
            quadNode& currChild = nodes[currNode.children[i]];
            totalMass += currChild.totalMass;
            weightedMass = weightedMass + currChild.centerOfMass * currChild.totalMass; 
        }
    }

    currNode.totalMass = totalMass;
    currNode.centerOfMass = weightedMass * (1/totalMass);
    return;
}

__global__ void calcAccel(body* bodies, quadNode* nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //global index
    if (idx >= N) return; //bounds check
    Vec3 locAcc(0, 0, 0); //local var to accumalte acceleration
    Vec3 locPos = bodies[idx].pos; //local position to avoid global reads
    int stack[64]; //A max stack size of 64 allows for (maxDepth * 3) + 1 <= 64, so 21 splits, meaning our minimum node size is 430 km
    int stackTop = 1; //variable to manage iterating through the stack
    stack[0] = 0; //setting the first item in stack to the idx of the root node

    while (stackTop > 0) {
        stackTop--;
        quadNode& currNode = nodes[stack[stackTop]];
        
        //Case 1, empty nodes
        if (currNode.totalMass == 0) continue; //Skipping empty nodes

        //Case 2, leaf nodes and internal nodes that satisfy theta
        if (currNode.bodyIdx != -1 || (currNode.size / (currNode.centerOfMass - locPos).magnitude()) <= theta_device) {
            if (currNode.bodyIdx == idx) continue; //the body skips itself

            double r = (currNode.centerOfMass - locPos).magnitude(); //distance between bodies
            double AccelMag = (G_device*currNode.totalMass)/(r*r + epsilon_device * epsilon_device); //I added a smoothing variable epsilon here, preventing the forces from exploding if the object gets to close to the central mass
            Vec3 dir((currNode.centerOfMass - locPos).uVec()); //direction from body idx to body i
            locAcc = locAcc + dir*AccelMag; //makes accel into a Vec3 and then accumalates it

            continue;
        }

        //Case 3, internal nodes that do not satisfy theta
        for (int i=0; i<4; i++) {
            if (currNode.children[i] != -1) {
                stack[stackTop++] = currNode.children[i];
            }
        }
    }

    bodies[idx].acc = locAcc;
}

/*
__global__ void calcAccelOld(body* bodies, int N) {
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
*/

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

void leapfrog(body* d_bodies, std::vector<body>& bodies, quadNode* d_nodes, std::vector<quadNode>& nodes, int& nodeCount, int numBlocks, int threadsPerBlock) {
    //Kick 1 + Drift and bring updated bodis back to cpu
        leapfrogPartOne<<<numBlocks, threadsPerBlock>>>(d_bodies, N);
        cudaDeviceSynchronize();
        cudaMemcpy(bodies.data(), d_bodies, N*sizeof(body), cudaMemcpyDeviceToHost);

        //Now logic to recalculate acceleration
        fillTree(bodies, nodes, nodeCount); //nodeCount's value right now is the nodes used in the previous tree, we pass that so fillTree knows how many nodes to reset.
        computeCOM(nodes, 0); //Always need to set the coms and masses after building the tree
        cudaMemcpy(d_nodes, nodes.data(), nodeCount * sizeof(quadNode), cudaMemcpyHostToDevice); //Here nodeCount represents the nodes used to build the current tree, so we only need to copy that amount of nodes over
        calcAccel<<<numBlocks, threadsPerBlock>>>(d_bodies, d_nodes); //now we actually run the function to calculate the new accelerations

        //Kick 2
        leapfrogPartTwo<<<numBlocks, threadsPerBlock>>>(d_bodies, N);
}

std::vector<body> initRandBodies(int N) {
    std::vector<body> bodies;

    std::mt19937 rng(23); //set seed for replicable results
    std::uniform_real_distribution<double> angleRange(0, 2 * Pi);
    std::uniform_real_distribution<double> radiusRange(1e6, maxRadius); //Everything spawns at least 1000 km away from the central mass

    body centralBody; //pos and vel is 0 by default
    centralBody.mass = 1e38; //very big black hole
    bodies.push_back(centralBody);

    for (int i=0; i<N-1; i++) { //our memcpy and malloc function copy exactly N, and with the central body we have N+1 bodies, so we just randomize N-1 bodies.
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
    //Init nodes vector
    std::vector<quadNode> nodes(maxNodes);
    //GPU prep
    int threadsPerBlock = 256;
    int numBlocks = (N+threadsPerBlock-1)/threadsPerBlock;
    
    //Prep for main logic
    //Bodies prep
    body* d_bodies;
    cudaMalloc(&d_bodies, N * sizeof(body)); //Reserve VRAM for bodies and nodes
    cudaMemcpy(d_bodies, bodies.data(), N*sizeof(body), cudaMemcpyHostToDevice); //Copies vector to GPU

    //Nodes Prep
    quadNode* d_nodes;
    int nodeCount = 0; //We create nodeCount outside fillTree so we only allocate space for the neccesary amount of nodes rather than the max each time. Its also set to 0 so the resetNodes loop works without error in the first treebuild.
    fillTree(bodies, nodes, nodeCount); //Fills the node tree with bodies
    computeCOM(nodes, 0); //sets the COMs for the nodes
    
    cudaMalloc(&d_nodes, maxNodes * sizeof(quadNode)); //allocates memory for maximum amt of nodes, meaning we can reuse it and save on malloc overhead
    cudaMemcpy(d_nodes, nodes.data(), nodeCount * sizeof(quadNode), cudaMemcpyHostToDevice); //copies filled node tree to GPU
    calcAccel<<<numBlocks, threadsPerBlock>>>(d_bodies, d_nodes); //assign initial acceleration to bodies

    //we dont need cudaDeviceSynchronize() here as the leapfrog CPU function is only made of kernels

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

        leapfrog(d_bodies, bodies, d_nodes, nodes, nodeCount, numBlocks, threadsPerBlock);

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
        window.display();
    }

    cudaFree(d_bodies);
    cudaFree(d_nodes);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return 0;
}
