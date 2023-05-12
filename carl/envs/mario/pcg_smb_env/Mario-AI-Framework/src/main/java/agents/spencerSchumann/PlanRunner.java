package agents.spencerSchumann;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * @author Spencer Schumann
 */
public class PlanRunner {

    private class Event {
        public int key;
        public boolean pressed;

        Event(int key, boolean pressed) {
            this.key = key;
            this.pressed = pressed;
        }
    }

    private int index;
    private int maxTime;
    private boolean[] action = new boolean[5];
    private HashMap<Integer, ArrayList<Event>> events;

    PlanRunner() {
        maxTime = -1;
        events = new HashMap<Integer, ArrayList<Event>>();
        rewind();
    }

    public boolean isDone() {
        return index > maxTime;
    }

    public boolean isLastAction() {
        return index == maxTime;
    }

    public int getIndex() {
        return index;
    }

    public int getLength() {
        return maxTime;
    }

    public void rewind() {
        index = 0;
    }

    public void addKey(int key) {
        addKey(key, 0);
    }

    public void addKey(int key, int timeStep) {
        addKeyEvent(key, timeStep, true);
    }

    public void addKey(int key, int timeStep, int duration) {
        addKeyEvent(key, timeStep, true);
        addKeyEvent(key, timeStep + duration, false);
    }

    private void addKeyEvent(int key, int timeStep, boolean pressed) {
        ArrayList<Event> keys = events.get(Integer.valueOf(timeStep));
        if (keys == null) {
            keys = new ArrayList<Event>();
            events.put(Integer.valueOf(timeStep), keys);
        }
        keys.add(new Event(key, pressed));
        maxTime = Math.max(maxTime, timeStep);
    }

    public boolean[] nextAction() {
        ArrayList<Event> keys = events.get(Integer.valueOf(index));
        if (keys != null) {
            for (Event e : keys) {
                action[e.key] = e.pressed;
            }
        }
        index++;
        return action;
    }
}
