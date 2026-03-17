#include <iostream>
#include <vector>
#include <cmath>
#include <SFML/Graphics.hpp>
#include <deque>

const double G = 6.67430e-11;
const double dt = 3600; //time per loop in seconds
const double AU = 1.496e11;
const double scale = 300.0/AU;

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

struct body {
    Vec3 pos;
    Vec3 vel;
    Vec3 acc;
    double mass;
};

void calcPairAcc(body& b1, body& b2) {
    double r = (b1.pos - b2.pos).magnitude(); //distance between the two bodies
    double m1 = b1.mass;
    double m2 = b2.mass;
    double forceMag = (G * m1 * m2)/(r*r); //gravitational force magnitude between bodies, using (Gm1m2/(r^2))
    Vec3 dir = (b2.pos - b1.pos).uVec(); //vector pointing from body 1 to body 2
    Vec3 forceVec = dir * forceMag; //multiplying direction vector from one planet to the other by the magnitude of the force to make it a vector
    b1.acc = b1.acc + forceVec * (1.0/b1.mass); //Acceleration is F/m as F=ma. Also we multiply by 1/mass because I didn't write a / operator.
    b2.acc = b2.acc - forceVec * (1.0/b2.mass); //The force vector for b2 is flipped due to Newton's third law
}

void calcAcc(std::vector<body>& bodies) {
    for (auto& b : bodies) { b.acc = Vec3(0, 0, 0); } //This goes through all our bodies and set the accelerations back to 0 as we need to recalculate them.
    for (int i=0; i<bodies.size(); i++) {
        for (int j=i+1; j<bodies.size(); j++) {
            calcPairAcc(bodies[i], bodies[j]); //Adjusting the acceleration of each body using the pair acceleration function on each unique pair.
        }
    }
}

void leapfrog(std::vector<body>& bodies) {
    for (auto& b : bodies) {
        b.vel = b.vel + b.acc*0.5*dt; //Kick 1, velocity updated to v_(i+1/2)
        b.pos = b.pos + b.vel*dt; //Drift, new position of bodies
    }
    calcAcc(bodies); //Updating acceleration of bodies to a_(i+1), needs to be outside loop
    for (auto& b : bodies) {
        b.vel = b.vel + b.acc*0.5*dt; //Kick 2, velocity updated to v_(i+1), check README to understand terminology used here
    }
}

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
    std::cout << "Started" << std::endl;
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

    calcAcc(bodies); //assign initial acceleration to bodies

    std::deque<Vec3> sunT;
    std::deque<Vec3> earthT;

    sf::RenderWindow window(sf::VideoMode({800, 800}), "nbody sim");
    std::cout << "window created" << std::endl; //Checking if the window got created, was having problems with dll files
    window.setFramerateLimit(240);

    while (window.isOpen()) {
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
        }

        leapfrog(bodies); // advance simulation each frame

        window.clear();
        
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

    return 0;
}