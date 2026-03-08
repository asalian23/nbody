#include <iostream>
#include <vector>
#include <cmath>

const double G = 6.67430e-11;
const double dt = 3600; //time per loop in seconds

struct Vec3 {
    double x, y, z;

    Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(double s)      const { return {x*s,   y*s,   z*s};   }

    double magnitude() const { return std::sqrt(x*x + y*y + z*z); }
    double dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }

    Vec3 uVec() const {
        double mag = magnitude();
        return {x/mag, y/mag, z/mag};
    }

};

struct Body {
    Vec3 pos;
    Vec3 vel;
    Vec3 acc;
    double mass;
};

void calcAcc(Body& b1, Body& b2){
    double r = (b1.pos - b2.pos).magnitude(); //distance between the two bodies
    double m1 = b1.mass;
    double m2 = b2.mass;
    double forceMag = (G * m1 * m2)/(r*r); //gravitational force magnitude between bodies, using (Gm1m2/(r^2))
    Vec3 dir = (b1.pos - b2.pos).uVec(); //vector pointing from body 1 to body 2
    Vec3 forceVec = dir * forceMag; //multiplying direction vector from one planet to the other by the magnitude of the force to make it a vector
    b1.acc = b1.acc + forceVec * (1.0/b1.mass); //Acceleration is F/m as F=ma. Also we multiply by 1/mass because I didn't write a / operator.
    b2.acc = b2.acc - forceVec * (1.0/b2.mass); //The force vector for b2 is flipped due to Newton's third law
}

int main() {
    Body earth;
    earth.mass = 5.972e24;
    earth.pos  = Vec3(1.496e11, 0, 0); // 1 AU from origin
    earth.vel  = Vec3(0, 29783, 0);    // orbital velocity m/s

    Body sun;
    sun.mass = 1.989e30;
    sun.pos  = Vec3(0, 0, 0); 
    sun.vel  = Vec3(0, 0, 0);

    std::cout << "Earth pos: " << earth.pos.x << ", " << earth.pos.y << "\n";
    std::cout << "Sun mass:  " << sun.mass << "\n";
    std::cout << "Everything compiled, how suprising.\n";

    return 0;
}