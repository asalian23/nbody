# We're approaching this problem using Leapfrog integration, which is a more accurate version of Euler's method, using the midpoint of the velocity over ∆t rather than the initial one. 

# We get v_(i+1/2) = v_(i-1/2) + a_i * ∆t and x_(i+1) = x_i + v_(i+1/2) * ∆t.
# Subbing the velocity equation into the position one we get x_f = x_i + v_i * t + (1/2) * a * t², the familiar kinematics formula from AP Physics 1.
# Also we get v_(i+1) = v_i + (1/2) * (a_i + a_(i+1)) * ∆t

# This formula remains 100% accurate as long as acceleration is constant, however that is never the case but it serves its purpose for approximation in this n-body simulator.

# We arrange the formulas into "Kick-Drift-Kick Form", split into three parts accordingly.

# Kick 1: v_(i+1/2) = v_i + (1/2) * a_i * ∆t — we "kick" the velocity up a half step.

# Drift: x_(i+1) = x_i + v_(i+1/2) * ∆t — we plug in our velocity from Kick 1 to find our new position.

# Kick 2: Using v_(i+1) = v_i + (1/2) * (a_i + a_(i+1)) * ∆t, distribute ∆t and (1/2), sub in Kick 1, and solve for final velocity using a_(i+1). This gets us v_(i+1) = v_(i+1/2) + (1/2) * a_(i+1) * ∆t .

# In a continuous loop, kick 1 and 2 can be merged into a single full step kick to half the number of needed velocity updates.

# Yoshida 4th Order Integrator
# After doing some more research, I've learned about another method to use in the n-body simulator
# Yoshida is 3x more intensive than leapfrog, but it will allow us to have larger timesteps with the same or better accuracy than leapfrog.
# Later on when we do CUDA I'll apply this, reducing the number of bodies we can have, but allowing us to simulate farther accurately with them.