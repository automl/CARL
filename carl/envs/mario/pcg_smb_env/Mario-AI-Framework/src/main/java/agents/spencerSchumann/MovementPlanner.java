/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package agents.spencerSchumann;

import java.util.ArrayList;
import java.util.Comparator;

import engine.helper.MarioActions;

import java.util.Collections;

/**
 * @author Spencer Schumann
 */
// Jump up from one floor to another, if possible.
public class MovementPlanner {

    Scene scene;
    MarioState mario;
    EnemySimulator enemySim;
    Edge targetFloor;

    public float[] projectedX;
    public float[] projectedY;

    MovementPlanner(Scene scene, MarioState mario, EnemySimulator enemySim) {
        this.scene = scene;
        this.mario = mario;
        this.enemySim = enemySim;
    }

    private int flightTimeForJump(int jumpTime, float height) {
        Scene simScene = new Scene(0, 0);
        simScene.floors.add(new Edge(-10.0f, height, 10.0f, height));
        MarioState simMario = new MarioState();
        simMario.mayJump = true;
        simMario.onGround = true;
        MotionSimulator sim = new MotionSimulator(simScene, simMario);

        int i;
        boolean[] jump = new boolean[5];
        jump[MarioActions.JUMP.getValue()] = true;
        for (i = 0; i < jumpTime; i++) {
            sim.update(jump);
        }
        boolean[] coast = new boolean[5];
        while (true) {
            sim.update(coast);
            if (sim.mario.onGround)
                return sim.getTicks();
            if (sim.mario.vy > 0.0f && sim.mario.y > height)
                return -1;
        }
    }

    private int ticksToPos(float pos) {
        Scene simScene = new Scene(0, 0);
        simScene.floors.add(new Edge(-1000.0f, 1.0f, pos, 1.0f));
        MarioState simMario = new MarioState();
        simMario.vx = mario.vx;
        MotionSimulator sim = new MotionSimulator(simScene, simMario);
        sim.leftWorldEdge = false;

        boolean[] run = new boolean[5];
        run[MarioActions.RIGHT.getValue()] = true;
        run[MarioActions.SPEED.getValue()] = true;
        while (true) {
            sim.update(run);
            if (sim.mario.x >= pos)
                return sim.getTicks();
        }
    }

    private float posFromTicks(int ticks) {
        Scene simScene = new Scene(0, 0);
        simScene.floors.add(new Edge(-1000.0f, 1.0f, ticks * 100.0f, 1.0f));  //TODO: 20.0f should be max speed
        MarioState simMario = new MarioState();
        simMario.vx = mario.vx;
        MotionSimulator sim = new MotionSimulator(simScene, simMario);
        sim.leftWorldEdge = false;

        int i;
        boolean[] run = new boolean[5];
        run[MarioActions.RIGHT.getValue()] = true;
        run[MarioActions.SPEED.getValue()] = true;
        for (i = 0; i < ticks; i++) {
            sim.update(run);
        }
        return sim.mario.x;
    }

    private boolean checkPlan(PlanRunner plan, Edge targetFloor) {
        MotionSimulator sim = new MotionSimulator(scene, mario);
        projectedX = new float[plan.getLength() + 1];
        projectedY = new float[plan.getLength() + 1];
        this.targetFloor = targetFloor;
        EnemySimulator es = enemySim.clone();
        while (!plan.isDone()) {
            es.update(scene);
            sim.update(plan.nextAction());
            projectedX[plan.getIndex() - 1] = sim.mario.x;
            projectedY[plan.getIndex() - 1] = sim.mario.y;
            for (Enemy e : es.enemies) {
                if (sim.mario.x >= (e.x - e.width / 2.0f - 4.0f) &&
                        sim.mario.x <= (e.x + e.width / 2.0f + 4.0f) &&
                        sim.mario.y >= e.y - e.height &&
                        sim.mario.y - sim.mario.height <= e.y) {
                    return false;
                }
            }
        }
        // Need one more to get the final movement...?
        //sim.update(plan.nextAction());
        plan.rewind();
        return sim.mario.x > targetFloor.x1 - 4.0f &&
                sim.mario.x < targetFloor.x2 + 4.0f &&
                sim.mario.y == targetFloor.y1 - 1.0f;
    }

    private PlanRunner planJump(Edge currentFloor, Edge targetFloor) {
        float ydiff = targetFloor.y1 - currentFloor.y1;
        // TODO: this is still a hack.  Should be -3, not +4, and time step shouldn't have the +1.
        // For some reason, jumps are coming up short at times.
        float xdiff = targetFloor.x1 - mario.x + 4.0f;
        if (xdiff < 0.0f)
            xdiff = targetFloor.x2 - mario.x - 3.0f;

        int ticks = ticksToPos(xdiff);
        //System.out.println(String.format("Time to travel %f: %d ticks", xdiff, ticks));
        int flightTime = 0;
        int jumpTime;
        for (jumpTime = 0; jumpTime <= 7; jumpTime++) {
            flightTime = flightTimeForJump(jumpTime, ydiff);
            if (flightTime < 0) {
                continue;
            }

            PlanRunner plan = new PlanRunner();
            // TODO: check to see if fight time will put Mario past right edge of target
            // TODO: check for collisions
            // TODO: jumps are falling short, find out why.

            //System.out.println(String.format("Planning on holding jump for %d ticks, total flight time %d ticks", jumpTime, flightTime));
            if (ticks <= flightTime) {
                //System.out.println(" ticks < flight time");
                // jump straight up, then move right later.
                plan.addKey(MarioActions.SPEED.getValue());
                plan.addKey(MarioActions.JUMP.getValue(), 0, jumpTime);
                plan.addKey(MarioActions.RIGHT.getValue(), flightTime - ticks, ticks);
            } else {
                int timeUntilJump = ticks - flightTime + 1;
                float posUntilJump = posFromTicks(timeUntilJump);
                if (posUntilJump + mario.x >= currentFloor.x2 + 4.0f) {
                    //System.out.println(" * Can't do it: not enough runway");
                    continue;
                }
                plan.addKey(MarioActions.SPEED.getValue());
                plan.addKey(MarioActions.RIGHT.getValue(), 0, ticks);
                plan.addKey(MarioActions.JUMP.getValue(), timeUntilJump, jumpTime);
            }

            // Check plan.  Does it successfully get to the target?
            if (checkPlan(plan, targetFloor))
                return plan;
            else
                continue;
        }
        return null;
    }

    class BestTarget implements Comparator<Edge> {
        public int compare(Edge o1, Edge o2) {
            if (o1.x2 > o2.x2)
                return -1;
            if (o1.x2 < o2.x2)
                return 1;
            if (o1.y1 > o2.y1)
                return -1;
            if (o1.y1 < o2.y1)
                return 1;
            return 0;
        }
    }

    private ArrayList<Edge> findTargetFloors(Edge currentFloor) {
        // Phase 1: find a floor that is above and to the right of the
        // current floor.  The nearest such floor is the target.
        ArrayList<Edge> targets = new ArrayList<Edge>();
        Collections.sort(scene.floors, new BestTarget());
        for (Edge e : scene.floors) {
            if (//(e.x2 > currentFloor.x2) &&
                    (e.x2 > mario.x) &&
                            //(e.y1 <= currentFloor.y1) &&
                            (e.y1 + 64.0f >= currentFloor.y1)) {
                //System.out.println(String.format("  Candidate target: (%f,%f)-(%f,%f)",
                //        e.x1, e.y1, e.x2, e.y2));
                targets.add(e);
            }
        }

        return targets;
    }

    private Edge findCurrentFloor() {
        for (Edge e : scene.floors) {
            if ((mario.y == e.y1 - 1) &&
                    (mario.x >= e.x1 - 4.0f) && // 4s account for Mario's width
                    (mario.x <= e.x2 + 4.0f)) {
                return e;
            }
        }
        return null;
    }

    public PlanRunner planMovement() {
        Edge currentFloor;
        ArrayList<Edge> targetFloors;
        currentFloor = findCurrentFloor();
        if (currentFloor != null) {
        } else {
            return null;
        }
        targetFloors = findTargetFloors(currentFloor);
        if (!targetFloors.isEmpty()) {
            for (Edge target : targetFloors) {
                if (target == currentFloor)
                    continue;
                PlanRunner plan = planJump(currentFloor, target);
                if (null != plan) {
                    // TODO: there may be several different plans that would work.
                    // If so, choose the best one.
                    return plan;
                }
            }
        }
        return null;
    }

    public Edge getTargetFloor() {
        return targetFloor;
    }
}
