#include <iostream>
#include <vector>
#include <cmath>
#include <deque>
#include <cuda_runtime.h>
#include <random>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

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
const int N = 75000; //number of bodies
//Do 2pir / v for the bodies to make sure the time step is good enough compared to the central mass
__constant__ float dt = 10; //time progressed per frame in seconds
__constant__ float epsilon_device = 4.65e-3f; //sun radius in AU
__constant__ float theta_device = 0.5f;

//const int maxDepth = 21; //ensures we don't have a stack overflow in the gpu
const int maxNodes = N*3; //worst case node usage, as its a sparse octree, each body can create a new node, then at worse each layer up contains half the nodes of the layer below, never exceed 2N, and then N was added for padding.
const float maxRadius = 6.0f;
const float Scale = 600.0f/maxRadius; //screen width some margins divided by the max radius
float centerPx = 1280.0f; //note that this should be adjusted by screen on the github
float centerPy = 800.0f;

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

struct octNode {
    Vec3 centerOfMass;
    Vec3 centerPos;
    float totalMass, size; 
    int children[8], bodyIdx, startIdx, endIdx; //start and end are present for leaf nodes with multiple bodies, due to having the same morton code
};

struct buildTask {
    int nodeIdx, start, end, depth;
};

__device__ int getOctant(unsigned int mCode, int depth) {
    return ((mCode >> (27 - 3*depth)) & 0b111); //extracts 3 bits of morton code corresponding to the current depth, which is the octant of the node at the depth in binary
}


__device__ void resetNode(octNode& node) {
    node.centerOfMass = Vec3();
    node.centerPos = Vec3();
    node.totalMass = 0;
    node.size = 0;
    node.bodyIdx = -1;
    node.startIdx = -1;
    node.endIdx = -1;
    for (int i=0; i<8; i++) node.children[i] = -1;
}

__global__ void splitLevel(unsigned int* codes, body* bodies, int* bodyIndices, octNode* nodes, buildTask* currLvlSplitQueue, int currQueueCount, buildTask* nextLvlSplitQueue, int* nextQueueCount, int* nodeCount) {
    int taskIdx = blockIdx.x * blockDim.x + threadIdx.x; //One thread executes one task (one split) in the current Queue
    if (taskIdx >= currQueueCount) return; //bounds check against total number of tasks in the current queue

    buildTask task = currLvlSplitQueue[taskIdx]; //Loads task to local memory
    int start = task.start, end = task.end, nodeIdx = task.nodeIdx, depth = task.depth; //range and depth, values needed to execute split
    octNode& parentNode = nodes[nodeIdx]; //loads the node being split
    
    //Note - start and end and indicies in the morton code array and the body indicies array, but not the nodes array
    //End is also one past the last index, so the range is [start, end)
    
    //Leaf node check
    if (end - start == 1 || depth >= 10 || codes[start] == codes[end-1]) { //if the range is 1, or if the depth is 10 or more, or if all the morton codes in the range are the same, then we have a leaf node
        //Note - The range can never be <1, unless we are running with N=0, but that's not a valid case
        //Same way, depth can never be >10, as 10 would already break it, just defenisve coding
        parentNode.bodyIdx = bodyIndices[start]; //assigns the body index to the parent node
        parentNode.totalMass = bodies[bodyIndices[start]].mass; //sets the mass of the leaf node to the one body
        parentNode.centerOfMass = bodies[bodyIndices[start]].pos; //sets the com of the leaf node to the one body
        
        parentNode.startIdx = start; //sets the start and end indicies of the leaf nodes range, this could occur if there are multiple bodies with the same morton code
        parentNode.endIdx = end;
        return;
    }

    int i = start; //variable i is functioning as local start, representing the first idx where the octant changes
    while (i < end) {
        int octant = getOctant(codes[i], depth); //gets the octant of the first body in range

        int runEnd = i; //init runEnd to start of run
        while (runEnd < end && getOctant(codes[runEnd], depth) == octant) { runEnd++; } //sets runEnd to the first index where the octant changes

        int childNodeIdx = atomicAdd(nodeCount, 1); //reserves the next avaliable nodeIdx for the child about to be created. This uses atomic add as multiple threads are trying to get slots in the nodes array at once to write into it.
        //Note - nodeCount is a varialbe tracking the number of nodes used and the next avaliable nodeIdx, and its incremented by one as a node is added
        parentNode.children[octant] = childNodeIdx; //sets the parent node's child pointer to the new child node

        octNode& childNode = nodes[childNodeIdx]; //loads child node to local memory
        resetNode(childNode); //inits childNode, also, this doesn't have <<<>>> as its a device function, not __global__, so doesn't need to be launched
        childNode.size = parentNode.size * 0.5f; //child node is half the size of the parent node
        float centerShift = parentNode.size * 0.25f; //center shifts by 1/4 parent size for child along each xyz with sign dependant on octant
        childNode.centerPos.x = parentNode.centerPos.x + centerShift * ((octant & 1) ? 1 : -1); //extracts the relevant bit according to convention and checks if its positive or 0, then changes the sign of centerShift accordinggly
        childNode.centerPos.y = parentNode.centerPos.y + centerShift * ((octant & 2) ? 1 : -1);
        childNode.centerPos.z = parentNode.centerPos.z + centerShift * ((octant & 4) ? 1 : -1);

        //Now the child node needs to be added to the nextLvlSplitQueue, so that it can be split in the next iteration of the loop
        int nextQueueIdx = atomicAdd(nextQueueCount, 1); //reserves the next avaliable index in the nextLvlSplitQueue, atomic as multiple threads are trying to add constructed child nodes to this at a time
        nextLvlSplitQueue[nextQueueIdx] = buildTask{childNodeIdx, i, runEnd, depth+1}; //adds the child node to the nextLvlSplitQueue with its range and depth
        
        i = runEnd; //sets i to the first index of the next octant run
    }

}

/*
octNode getRootNode(const std::vector<body>& bodies) {
    float xMin = bodies[0].pos.x;
    float xMax = bodies[0].pos.x;
    float yMin = bodies[0].pos.y;
    float yMax = bodies[0].pos.y;
    float zMin = bodies[0].pos.z;
    float zMax = bodies[0].pos.z;

    for (int i=0; i<bodies.size(); i++) {
        xMin = std::min(xMin, bodies[i].pos.x);
        xMax = std::max(xMax, bodies[i].pos.x);
        yMin = std::min(yMin, bodies[i].pos.y);
        yMax = std::max(yMax, bodies[i].pos.y);
        zMin = std::min(zMin, bodies[i].pos.z);
        zMax = std::max(zMax, bodies[i].pos.z);
    }

    octNode root;
    resetNode(root);
    root.size = std::max({xMax-xMin, yMax-yMin, zMax-zMin}) * 1.01; //Multiplying by 1.01 handles rare edge cases
    root.centerPos = Vec3((xMin + xMax)/2.0, (yMin + yMax)/2.0, (zMin + zMax)/2.0);
    
    return root;
}
*/

__device__ unsigned int spreadBits(unsigned int v) {
    unsigned int result = 0;
    for (int i = 0; i < 10; i++) {
        result |= ((v >> i) & 1u) << (i * 3);   // bit i goes to position i*3
    }
    return result;
}

//Note - morton3D sets the convention, the bits are ordered in zyx, 4s being z, 2s being y, and 1s being x
__device__ unsigned int morton3D(unsigned int x, unsigned int y, unsigned int z) {
    return (spreadBits(z) << 2) | (spreadBits(y) << 1) | spreadBits(x); //combine spread xyz values into morton coord, | is just bitwise or while || is logical or
}

__global__ void mortonEncode(body* bodies, unsigned int* codes, int* bodyIndices, Vec3 minCoord, float inversedRange) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    Vec3 pos = bodies[idx].pos; //pull pos to register for one global mem read
    
    unsigned int x = (unsigned int)((pos.x - minCoord.x) * inversedRange * 1023.0f); //normalizes the position then scales it to a 10 bit range, then its truncated to an int
    unsigned int y = (unsigned int)((pos.y - minCoord.y) * inversedRange * 1023.0f);
    unsigned int z = (unsigned int)((pos.z - minCoord.z) * inversedRange * 1023.0f);
    //note, this breaks if we have a stray body well outside the visible range, as the "cell sizes", the smallest difference the encoding can differentiate, varies by the bounding size. Will implement culling of stray bodies to counter.

    x = min(x, 1023u); //float rounding can result in 1024 so we bind the value to 1023 (1024 would require 11 bits)
    y = min(y, 1023u);
    z = min(z, 1023u);

    codes[idx] = morton3D(x, y, z); //takes the normalized values for xyz and interleaves the bits to form the morton code
    bodyIndices[idx] = idx; //Fills the array of bodyIndices with its current idx value, this will be sorted the same way as the codes array so accessing the same element of both arrays will tell us which morton code corresponds to which body
}

/*
int getOctant(const body& b, const octNode& node) {
    return (b.pos.x >= node.centerPos.x) | ((b.pos.y >= node.centerPos.y) << 1) | ((b.pos.z >= node.centerPos.z) << 2); //using bitwise to avoid branching
}
*/

void calcChildCenterPos (const octNode& parent, const int& octant, Vec3& childCenterPos) {
    int xBit = octant & 1; //extract bits, the & operator only keeps the int n bits from the left
    int yBit = (octant >> 1) & 1;
    int zBit = (octant >> 2) & 1;
    float shift = parent.size /4.0; //shift is 1/4 of parent node width

    //2*bit - 1, will turn the 0 or 1 into -1 or 1 nicely to apply shift
    childCenterPos.x = parent.centerPos.x + (2 * xBit - 1) * shift;
    childCenterPos.y = parent.centerPos.y + (2 * yBit - 1) * shift;
    childCenterPos.z = parent.centerPos.z + (2 * zBit - 1) * shift;
}

/*
//The next function also builds the tree. Note - This is a sparse tree, so watch out for nonexistent nodes when recursing
void insertBody(std::vector<body>& bodies, std::vector<octNode>& nodes, int& nodeCount, int bodyIdx, int nodeIdx, int currDepth) { //This function must be seperate as it recurses into itself
    const body& currBody = bodies[bodyIdx];
    octNode& currNode = nodes[nodeIdx];

    bool hasChildren = (currNode.children[0] != -1 || currNode.children[1] != -1 || currNode.children[2] != -1 || currNode.children[3] != -1 || currNode.children[4] != -1 || currNode.children[5] != -1 || currNode.children[6] != -1 || currNode.children[7] != -1);

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
        int prevBodyOctant = getOctant(bodies[prevBodyIdx], currNode); //prev body octant used to init the space for the prev body
        currNode.bodyIdx = -1;
        
        //The following logic is to readd the previous body to the appropriate child node
        octNode& currChildNode = nodes[nodeCount]; //sets the childnodes location to next avaliable spot in nodes array
        currNode.children[prevBodyOctant] = nodeCount; //idx of child node is next avaliable nodeIdx
        resetNode(currChildNode); //init child node
        currChildNode.size = currNode.size / 2.0; //sets child size
        calcChildCenterPos(currNode, prevBodyOctant, currChildNode.centerPos); //sets child center
        nodeCount++;

        //For the following insertion we are still passing the parent's nodeIdx to let case 3 handle recursing into the child nodes, and as such we do not increment the depth
        insertBody(bodies, nodes, nodeCount, prevBodyIdx, nodeIdx, currDepth);
        //Not returning here like in the dense tree, for the sparse tree we don't know which node needs to be initialized for the new body, so we let case 3 handle that and let the parent node naturally fall into it.
    }

    //Third Case, internal node
    int octant = getOctant(currBody, currNode);

    if (currNode.children[octant] == -1) {
        octNode& currChildNode = nodes[nodeCount]; //sets the childnodes location to next avaliable spot in nodes array
        currNode.children[octant] = nodeCount; //idx of child node is next avaliable nodeIdx
        resetNode(currChildNode); //init child node
        currChildNode.size = currNode.size / 2.0; //sets child size
        calcChildCenterPos(currNode, octant, currChildNode.centerPos); //sets child center
        nodeCount++;
    }
    insertBody(bodies, nodes, nodeCount, bodyIdx, currNode.children[octant], currDepth+1);
}
*/

/*
void fillTree(std::vector<body>& bodies, std::vector<octNode>& nodes, int& nodeCount) { //loops through bodies and inserts each one
    for (int i=0; i<nodeCount; i++) resetNode(nodes[i]); //resets the old nodes

    nodeCount = 1; //the root node, nodeCount represents both the number of node's allocated and the next avalaible nodeIdx
    nodes[0] = getRootNode(bodies);

    for (int bodyIdx=0; bodyIdx<N; bodyIdx++) {
        insertBody(bodies, nodes, nodeCount, bodyIdx, 0, 0);
    }
}
*/

/*
void computeCOM(std::vector<octNode>& nodes, int nodeIdx) { //nodeIdx is just read not changed so we don't pass by reference
    octNode& currNode = nodes[nodeIdx];

    bool hasChildren = (currNode.children[0] != -1 || currNode.children[1] != -1 || currNode.children[2] != -1 || currNode.children[3] != -1 || currNode.children[4] != -1 || currNode.children[5] != -1 || currNode.children[6] != -1 || currNode.children[7] != -1);
    
    //Case 1, empty node
    if (currNode.bodyIdx == -1 && !hasChildren) {
        return;
    }

    //Case 2, leaf node
    if (currNode.bodyIdx != -1) {
        return; //we also don't do anything here as its com and total mass are set correctly already in the insertBody function
    }

    //Case 3, internal node
    float totalMass = 0.0;
    Vec3 weightedMass;
    for (int i=0; i<8; i++) { //go through each child
        if (currNode.children[i] != -1) { //will cause error trying to acces nodes[-1] if the child doesn't exist because sparse trees don't initialize empty nodes
            computeCOM(nodes, currNode.children[i]); //recurse down to compute com for each child, and then those recurse and so on, making this a bottom up search
            
            octNode& currChild = nodes[currNode.children[i]];
            totalMass += currChild.totalMass;
            weightedMass = weightedMass + currChild.centerOfMass * currChild.totalMass; 
        }
    }

    currNode.totalMass = totalMass;
    currNode.centerOfMass = weightedMass * (1/totalMass);
    return;
}
*/

__global__ void calcCOMByLvl(octNode* nodes, body* bodies, int* bodyIndices, int lvlStart, int lvlEnd) {
    int idx = lvlStart + blockIdx.x * blockDim.x + threadIdx.x; //starts the thread indicies at the start of the level
    if (idx >= lvlEnd) return; //stop if the thread is out of bounds for the current level

    //Leaf Node with one body
    if (nodes[idx].bodyIdx != -1 && nodes[idx].endIdx - nodes[idx].startIdx == 1) { //If the node is a leaf that has one body, checked by subtracted start and end indicies because if it was one body the difference should be 1
        return; //Already taken care of in splitLvl
    }
    
    //Leaf Node with multiple bodies
    if (nodes[idx].bodyIdx != -1 && nodes[idx].endIdx - nodes[idx] .startIdx != 1) {
        float totalMass = 0.0f;
        Vec3 weightedMass(0, 0, 0);
        for (int i=nodes[idx].startIdx; i<nodes[idx].endIdx; i++) {
            body& b = bodies[bodyIndices[i]]; //gets the actual body
            totalMass += b.mass; //the masses are summed
            weightedMass = weightedMass + b.pos * b.mass; //the coms are summed and weighted by the bodies mass
        }
        nodes[idx].totalMass = totalMass; //sets the leaf's mass
        nodes[idx].centerOfMass = weightedMass * (1.0f/totalMass); //divides by the total mass to get the leaf's accurate com
    }

    //Internal
    if (nodes[idx].bodyIdx == -1) {
        float totalMass = 0.0f;
        Vec3 weightedMass(0, 0, 0);
        for (int i=0; i<8; i++) {
            if (nodes[idx].children[i] != -1) { //if the child exists
                octNode& childNode = nodes[nodes[idx].children[i]];
                totalMass += childNode.totalMass; //the masses are summed
                weightedMass = weightedMass + childNode.centerOfMass * childNode.totalMass; //the coms are summed and weighted by the child's mass
            }
        }
        nodes[idx].totalMass = totalMass; //sets the internal node's mass
        nodes[idx].centerOfMass = weightedMass * (1.0f/totalMass); //divides by the total mass to get the internal node's accurate com
    }

}

void buildTreeGPU(unsigned int* d_codes, body* d_bodies, int* d_bodyIndices, octNode* d_nodes, buildTask* d_currQueue, buildTask* d_nextQueue, int* d_nextQueueCount, int* d_nodeCount, int threadsPerBlock, float mortonRange, std::vector<int>& treeDepthBounds) {
    octNode rootNode;
    rootNode.centerPos = Vec3(0, 0, 0); //center of the root node is at the origin
    rootNode.size = 2 * mortonRange;
    rootNode.totalMass = 0; //will be set up com pass
    rootNode.centerOfMass = Vec3(0, 0, 0); //will be set up com pass
    rootNode.bodyIdx = -1; //root node is not a leaf node
    for (int i=0; i<8; i++) rootNode.children[i] = -1; //root node has no children yet
    cudaMemcpy(d_nodes, &rootNode, sizeof(octNode), cudaMemcpyHostToDevice); //copy root node to device



    int h_nodeCount = 1; //start with root node
    cudaMemcpy(d_nodeCount, &h_nodeCount, sizeof(int), cudaMemcpyHostToDevice); //reset d_nodeCount to 1, otherwise it would keep growing every frame and overflow the maxNode limit

    buildTask rootTask = {0, 0, N, 0}; //root node task, nodeIdx=0, start=0, end=N, depth=0
    cudaMemcpy(d_currQueue, &rootTask, sizeof(buildTask), cudaMemcpyHostToDevice); //copy root task to device
    int currCount = 1; //One task (root) in the current queue

    std::fill(treeDepthBounds.begin(), treeDepthBounds.end(), 0); //zero the vector so the last frames bounds don't interfere with this ones
    int currentDepth = 0; //tracks the current depth of the tree being built
    treeDepthBounds[0] = h_nodeCount; //sets the depth bound for the root node, as the loop splits the rootNode in the first pass, so the first recorded value there is the end of level 1, not 0

    while (currCount > 0) {
        int zero = 0; //need this as we can't directly set the value of d_nextQueueCount as its on the GPU
        cudaMemcpy(d_nextQueueCount, &zero, sizeof(int), cudaMemcpyHostToDevice); //reset next queue count to 0
        
        int blocksNeeded = (currCount + threadsPerBlock - 1) / threadsPerBlock;
        splitLevel<<<blocksNeeded, threadsPerBlock>>>(d_codes, d_bodies, d_bodyIndices, d_nodes, d_currQueue, currCount, d_nextQueue, d_nextQueueCount, d_nodeCount);
        cudaMemcpy(&currCount, d_nextQueueCount, sizeof(int), cudaMemcpyDeviceToHost); //update currCount with the number of tasks in the next queue, swapping that value
        //Its possible here that we finish early, ie splitLevel finds every node to be a leaf and doesnt create any more child nodes, in this case we are done but their may be the same nodeCount written again into the next level, watch out for this in the com computation

        //Swap queues for next level
        buildTask* temp = d_currQueue;
        d_currQueue = d_nextQueue;
        d_nextQueue = temp;

        currentDepth++; //increment depth for next level
        cudaMemcpy(&h_nodeCount, d_nodeCount, sizeof(int), cudaMemcpyDeviceToHost); //update h_nodeCount with the number of nodes used in the tree so far
        treeDepthBounds[currentDepth] = h_nodeCount; //set the depth bound to the nextNodeIdx

    }
}

void calcCOM(octNode* nodes, body* bodies, int* bodyIndices, std::vector<int>& treeDepthBounds, int threadsPerBlock) {
    
    //retrieves the deepest level the tree is built till, as it could be more shallow than 10
    int deepestLevel = 0;
    while (treeDepthBounds[deepestLevel] != 0 && deepestLevel < treeDepthBounds.size()) deepestLevel++;
    deepestLevel--;
    
    //Test printing the values in treeDepthBounds and deepestLevel
    //for (int k = 0; k < 11; k++) printf("bounds[%d]=%d ", k, treeDepthBounds[k]);
    //printf("\ndeepestLevel=%d\n", deepestLevel);

    for (int i = deepestLevel; i>=0; i--) {
        int lvlStart;
        if (i!= 0) {
            lvlStart = treeDepthBounds[i-1];
        } 
        else {
            lvlStart = 0;
        }
        int lvlEnd = treeDepthBounds[i];

        if (lvlEnd - lvlStart == 0) continue;
        int blocksRequired = ((lvlEnd - lvlStart) + (threadsPerBlock - 1)) / threadsPerBlock;
        calcCOMByLvl<<<blocksRequired, threadsPerBlock>>>(nodes, bodies, bodyIndices, lvlStart, lvlEnd);
        cudaDeviceSynchronize();
    }
}

__global__ void calcAccel(body* bodies, octNode* nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //global index
    if (idx >= N) return; //bounds check
    Vec3 locAcc(0, 0, 0); //local var to accumalte acceleration
    Vec3 locPos = bodies[idx].pos; //local position to avoid global reads
    int stack[148]; //A max stack size of 128 allows for (maxDepth * 7) + 1 <= 148, so 21 splits, meaning our minimum node size is 430 km (not sure on the exact num actually)
    int stackTop = 1; //variable to manage iterating through the stack
    stack[0] = 0; //setting the first item in stack to the idx of the root node

    while (stackTop > 0) {
        stackTop--;
        octNode& currNode = nodes[stack[stackTop]];
        
        //Case 1, empty nodes
        if (currNode.totalMass == 0) continue; //Skipping empty nodes

        //Case 2, leaf nodes and internal nodes that satisfy theta
        if (currNode.bodyIdx != -1 || (currNode.size / (currNode.centerOfMass - locPos).magnitude()) <= theta_device) {
            if (currNode.bodyIdx == idx) continue; //the body skips itself

            Vec3 rVec = (currNode.centerOfMass - locPos); //vector between bodies
            float rSmoothedSquared = rVec.dot(rVec) + epsilon_device * epsilon_device; //So because we're using floats, bodies can be overlapping each other makes rVec zero, meaning we can't run magnitude due to NaN poisoning
            float rSmoothedInverse = rsqrt(rSmoothedSquared);
            //float AccelMag = (G_device*currNode.totalMass)/(rSmoothedSquared); //I added a smoothing variable epsilon here, preventing the forces from exploding if the object gets to close to the central mass
            //Vec3 dir(rVec * rSmoothedInverse); //direction from body idx to body i, same problem here with uVec, NaN poisoning as it also uses magnitude function in uVec. Replaced with new logic to get direction.
            locAcc = locAcc + (rVec * rSmoothedInverse) * G_device * currNode.totalMass * rSmoothedInverse * rSmoothedInverse; //makes accel into a Vec3 and then accumalates it

            continue;
        }

        //Case 3, internal nodes that do not satisfy theta
        for (int i=0; i<8; i++) {
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
            float r = (tile[i].pos - locPos).magnitude(); //distance between bodies
            float AccelMag = (G_device*tile[i].mass)/(r*r + epsilon_device * epsilon_device); //I added a smoothing variable epsilon here, preventing the forces from exploding if the object gets to close to the central mass
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

void leapfrog(body* d_bodies, octNode* d_nodes, int* d_nodeCount, buildTask* d_currQueue, buildTask* d_nextQueue, int* d_nextQueueCount, int numBlocks, int threadsPerBlock, unsigned int* d_codes, int* d_bodyIndices, Vec3 mortonMin, float inversedRange, float mortonRange, std::vector<int>& treeDepthBounds) {
    //Kick 1 + Drift and bring updated bodies back to cpu
        leapfrogPartOne<<<numBlocks, threadsPerBlock>>>(d_bodies, N);
        
       /* 
        cudaMemcpy(bodies.data(), d_bodies, N*sizeof(body), cudaMemcpyDeviceToHost);
        //Now logic to recalculate acceleration
        fillTree(bodies, nodes, nodeCount); //nodeCount's value right now is the nodes used in the previous tree, we pass that so fillTree knows how many nodes to reset.
        computeCOM(nodes, 0); //Always need to set the coms and masses after building the tree
        cudaMemcpy(d_nodes, nodes.data(), nodeCount * sizeof(octNode), cudaMemcpyHostToDevice); //Here nodeCount represents the nodes used to build the current tree, so we only need to copy that amount of nodes over
        calcAccel<<<numBlocks, threadsPerBlock>>>(d_bodies, d_nodes); //now we actually run the function to calculate the new accelerations
        */

        //Create the morton codes for the bodies and the body indicies array and sort them both together according to the codes
        mortonEncode<<<numBlocks, threadsPerBlock>>>(d_bodies, d_codes, d_bodyIndices, mortonMin, inversedRange); 
        thrust::device_ptr<unsigned int> codesPtr(d_codes);
        thrust::device_ptr<int> indicesPtr(d_bodyIndices);
        thrust::sort_by_key(codesPtr, codesPtr + N, indicesPtr);
        
        buildTreeGPU(d_codes, d_bodies, d_bodyIndices, d_nodes, d_currQueue, d_nextQueue, d_nextQueueCount, d_nodeCount, threadsPerBlock, mortonRange, treeDepthBounds); //builds the tree on the GPU
        calcCOM(d_nodes, d_bodies, d_bodyIndices, treeDepthBounds, threadsPerBlock);
        calcAccel<<<numBlocks, threadsPerBlock>>>(d_bodies, d_nodes);

        //Kick 2
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
    unsigned int* d_codes; //unsigned as the morton codes can never be negative, although no difference for 32 bits either way
    int* d_bodyIndices; //Body indicies array that will be sorted in the same way as the morton codes, so the morton code and index 1 has a its corresponding bodyIdx defined in this array's index 1
    buildTask* d_currTaskQueue; //Ptr to the current task queue, or queue of the current splits that need to occur
    buildTask* d_nextTaskQueue; //Ptr to the next task queue, or queue of the splits that need to occur at the next depth level
    int* d_nextQueueCount; //Ptr to the nextQueueCount variable, which tracks the number of tasks in the nextTaskQueue
    int* d_nodeCount; //Ptr to the nodeCount variable, which is used to track the next avaliable nodeIdx and the number of nodes used

    cudaMalloc(&d_bodies, N * sizeof(body)); //Reserve VRAM for bodies
    cudaMalloc(&d_codes, N *sizeof(unsigned int)); //Reserves VRAM for morton codes
    cudaMalloc(&d_bodyIndices, N *sizeof(int)); //Reserves VRAM for idx array for bodyIdx that morton code corresponds to
    
    //Both these queues are allocated size N as the maximum number of tasks that can be queued is N, one for each body
    cudaMalloc(&d_currTaskQueue, N * sizeof(buildTask)); //Reserves VRAM for current task queue
    cudaMalloc(&d_nextTaskQueue, N * sizeof(buildTask)); //Reserves VRAM for next task queue
    
    cudaMalloc(&d_nextQueueCount, sizeof(int)); //Reserves VRAM for one int, the next queue count
    cudaMalloc(&d_nodeCount, sizeof(int)); //Reserves VRAM for one int, the node count

    cudaMemcpy(d_bodies, bodies.data(), N*sizeof(body), cudaMemcpyHostToDevice); //Copies vector to GPU

    //Nodes Prep
    octNode* d_nodes;
    std::vector<int> treeDepthBounds(11); //Holds the nodeIdx that is the end of each depth or start of the next depth, for example, treeDepthBounds[0] is the end of depth 0, or 1, also the start of depth 1
    //int nodeCount = 0; Not needed on GPU
    //fillTree(bodies, nodes, nodeCount); //Fills the node tree with bodies
    //computeCOM(nodes, 0); //sets the COMs for the nodes
    
    cudaMalloc(&d_nodes, maxNodes * sizeof(octNode)); //allocates memory for maximum amt of nodes, meaning we can reuse it and save on malloc overhead
    //cudaMemcpy(d_nodes, nodes.data(), nodeCount * sizeof(octNode), cudaMemcpyHostToDevice); //On GPU no longer needed

    //morton code prep
    const float mortonRange = 20.0f; //covers all of the screen and a bit more
    Vec3 mortonMin(-mortonRange, -mortonRange, -mortonRange);
    float inversedRange = 1.0f/ (2* mortonRange);

    //Initial tree build and initial acceleration calculation (needs morton codes init first)
    mortonEncode<<<numBlocks, threadsPerBlock>>>(d_bodies, d_codes, d_bodyIndices, mortonMin, inversedRange);
    thrust::device_ptr<unsigned int> codesPtr(d_codes);
    thrust::device_ptr<int> indicesPtr(d_bodyIndices);
    thrust::sort_by_key(codesPtr, codesPtr + N, indicesPtr);
    
    buildTreeGPU(d_codes, d_bodies, d_bodyIndices, d_nodes, d_currTaskQueue, d_nextTaskQueue, d_nextQueueCount, d_nodeCount, threadsPerBlock, mortonRange, treeDepthBounds); //Initial tree build
    
    //Test for if BuildTreeGPU is working by checking the nodes created for the first tree build
    int h_nc;
    cudaMemcpy(&h_nc, d_nodeCount, sizeof(int), cudaMemcpyDeviceToHost);
    printf("node count: %d\n", h_nc);

    calcCOM(d_nodes, d_bodies, d_bodyIndices, treeDepthBounds, threadsPerBlock);
    
    octNode h_root;
    cudaMemcpy(&h_root, d_nodes, sizeof(octNode), cudaMemcpyDeviceToHost);
    printf("root mass: %f, com: (%f,%f,%f)\n", h_root.totalMass,
    h_root.centerOfMass.x, h_root.centerOfMass.y, h_root.centerOfMass.z);
    fflush(stdout);
    
    calcAccel<<<numBlocks, threadsPerBlock>>>(d_bodies, d_nodes); //assign initial acceleration to bodies
    
    //Creating window and loading OpenGL Functions
    glfwInit(); //starts library
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); //Setting version to 4.6 for GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6); 
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); //Selects Core profile
    GLFWwindow* window = glfwCreateWindow(2560, 1600, "nbody sim", NULL, NULL); //creating the window, the first NULL is for passing a monitor and the second NULL is for sharing openGl over mutliple windows
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

            leapfrog(d_bodies, d_nodes, d_nodeCount, d_currTaskQueue, d_nextTaskQueue, d_nextQueueCount, numBlocks, threadsPerBlock, d_codes, d_bodyIndices, mortonMin, inversedRange, mortonRange, treeDepthBounds); //increment pos and vel, and updates acceleration
            
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

            writeToVBO<<<numBlocks, threadsPerBlock>>>(d_bodies, d_vboPtr, cosX, sinX, cosY, sinY, camDist, Scale, centerPx, centerPy, centerPx, centerPy); //actually write the position data into the buffer, //its center px x and px y because its the same as half the width and height

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
    cudaFree(d_nodes);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return 0;
}
