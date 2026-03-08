#include <iostream>
#include <vector>
#include <cmath>

const double G = 6.67430e-11;

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

double calcG(Body b1, Body b2){
    double r = (b1.pos - b2.pos).magnitude(); //distance between the two bodies
    double m1 = b1.mass;
    double m2 = b2.mass;
    return (G * m1 * m2)/(r*r); //gravitational force between bodies, using (Gm1m2/(r^2))
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
    std::cout << "It compiled. You're good.\n";

    return 0;
}