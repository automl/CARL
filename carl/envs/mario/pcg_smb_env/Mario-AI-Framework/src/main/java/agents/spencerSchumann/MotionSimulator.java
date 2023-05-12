/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package agents.spencerSchumann;

import engine.helper.MarioActions;

/**
 * @author Spencer Schumann
 */
public class MotionSimulator {

    private Scene scene;
    public MarioState mario;
    public boolean collision;
    private int ticks;
    public boolean leftWorldEdge = true;

    public MotionSimulator(Scene scene, MarioState mario) {
        this.scene = scene.clone();
        this.mario = mario.clone();
    }

    public void updateScene(Scene scene) {
        this.scene.update(scene);
    }

    private void handleHorizontalInput(boolean[] action) {
        float xSpeed = action[MarioActions.SPEED.getValue()] ? 1.2f : 0.6f;
        if (action[MarioActions.LEFT.getValue()])
            mario.vx -= xSpeed;
        if (action[MarioActions.RIGHT.getValue()])
            mario.vx += xSpeed;
    }

    // TODO: simulator known issues:
    // 1. Simulator sometimes allows jump when it shouldn't: when running off of
    //    edges, and when repeatedly pressing the jump button

    private void handleJumpInput(boolean[] action) {
        mario.vy *= 0.85f;
        if (!mario.onGround)
            mario.vy += 3.0f;

        // TODO: instead of all this boolean[] nonsense, create an Action class with jump etc. fields
        if (!action[MarioActions.JUMP.getValue()]) {
            mario.jumpTime = 0;
            if (mario.onGround)
                mario.mayJump = true;
        }

        if (action[MarioActions.JUMP.getValue()]) {
            if (mario.jumpTime > 0) {
                mario.vy = -1.9f * mario.jumpTime--;
            } else if (mario.mayJump) {
                mario.jumpTime = 7;
                mario.vy = -1.9f * mario.jumpTime;
            }
            mario.mayJump = false;
        }
    }

    // Applies the given action to run one simulation time step
    public void update(boolean[] action) {
        handleHorizontalInput(action);
        handleJumpInput(action);

        moveHorizontally();
        moveVertically();

        ticks++;
    }

    // Add using the goofy 8 at a time method used in Mario.java.
    // Skipping this step causes small errors due to floating point inaccuracy.
    private float goofyAdd(float a, float b) {
        while (b > 8.0f) {
            b -= 8.0f;
            a += 8.0f;
        }
        while (b < -8.0f) {
            b += 8.0f;
            a -= 8.0f;
        }
        return a + b;
    }

    // Move horizontally, checking for collisions
    private void moveHorizontally() {
        if (Math.abs(mario.vx) < 0.5f)
            mario.vx = 0.0f;

        float newX = goofyAdd(mario.x, mario.vx);
        // Is there a wall between x & newX?
        for (Edge e : scene.walls) {
            //System.out.println(String.format("Pos:(%f,%f)  New X: %f  Wall:(%f,%f)@%f", scene.pos.x, scene.pos.y, newX, e.y1, e.y2, e.x1));
            // In the right Y range?
            // TODO: I think the Y check is off a bit, especially the y2 portion.
            if (e.y1 <= mario.y &&
                    e.y2 >= mario.y - mario.height) {
                // Collision going right?
                if (mario.x + 4.0f <= e.x1 && e.x1 <= newX + 4.0f) {
                    mario.x = e.x1 - 5.0f;
                    mario.vx = 0.0f;
                    collision = true;
                    return;
                }
                // Collision going left?
                if (newX - 4.0f <= e.x1 && e.x1 <= mario.x - 4.0f) {
                    mario.x = e.x1 + 4.0f;
                    mario.vx = 0.0f;
                    collision = true;
                    return;
                }
            }
        }
        mario.x = newX;
        mario.vx *= 0.89f;

        if (leftWorldEdge && mario.x < 0.0f) {
            mario.x = 0.0f;
            mario.vx = 0.0f;
        }
    }

    // Move vertically, checking for collisions
    private void moveVertically() {
        float newY = goofyAdd(mario.y, mario.vy);
        // Check for floor
        if (mario.vy >= 0.0f) {
            for (Edge e : scene.floors) {
                // In right X range and has Y intersect?
                if (e.x1 < mario.x + 4.0f &&
                        e.x2 > mario.x - 4.0f &&
                        mario.y <= e.y1 && e.y1 - 1.0f <= newY) {
                    mario.y = e.y1 - 1.0f;
                    mario.onGround = true;
                    mario.jumpTime = 0;
                    return;
                }
            }
        }
        // Check for ceiling
        else if (mario.vy < 0.0f) {
            for (Edge e : scene.ceilings) {
                // In right X range and has Y intersect?
                if (e.x1 < mario.x + 4.0f &&
                        e.x2 > mario.x - 4.0f &&
                        mario.y - mario.height >= e.y1 &&
                        e.y1 >= newY - mario.height) {
                    mario.y = e.y1 + mario.height;
                    mario.vy = 0.0f;
                    mario.jumpTime = 0;
                    collision = true;
                    return;
                }
            }
        }
        mario.onGround = false;
        mario.mayJump = false;
        mario.y = newY;
    }

    public Scene getScene() {
        return scene;
    }

    public float getX() {
        return mario.x;
    }

    public void setX(float x) {
        mario.x = x;
    }

    public float getVX() {
        return mario.vx;
    }

    public float getY() {
        return mario.y;
    }

    public void setY(float y) {
        mario.y = y;
    }

    public int getTicks() {
        return ticks;
    }
}
