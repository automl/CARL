package agents.robinBaumgarten;

import java.util.ArrayList;

import engine.core.MarioForwardModel;
import engine.core.MarioTimer;
import engine.helper.GameStatus;

public class AStarTree {
    public SearchNode bestPosition;
    public SearchNode furthestPosition;
    float currentSearchStartingMarioXPos;
    ArrayList<SearchNode> posPool;
    ArrayList<int[]> visitedStates = new ArrayList<int[]>();
    private boolean requireReplanning = false;

    private ArrayList<boolean[]> currentActionPlan;
    int ticksBeforeReplanning = 0;

    private MarioForwardModel search(MarioTimer timer) {
        SearchNode current = bestPosition;
        boolean currentGood = false;
        int maxRight = 176;
        while (posPool.size() != 0
                && ((bestPosition.sceneSnapshot.getMarioFloatPos()[0] - currentSearchStartingMarioXPos < maxRight) || !currentGood)
                && timer.getRemainingTime() > 0) {
            current = pickBestPos(posPool);
            if (current == null) {
                return null;
            }
            currentGood = false;
            float realRemainingTime = current.simulatePos();

            if (realRemainingTime < 0) {
                continue;
            } else if (!current.isInVisitedList && isInVisited((int) current.sceneSnapshot.getMarioFloatPos()[0],
                    (int) current.sceneSnapshot.getMarioFloatPos()[1], current.timeElapsed)) {
                realRemainingTime += Helper.visitedListPenalty;
                current.isInVisitedList = true;
                current.remainingTime = realRemainingTime;
                current.remainingTimeEstimated = realRemainingTime;
                posPool.add(current);
            } else if (realRemainingTime - current.remainingTimeEstimated > 0.1) {
                // current item is not as good as anticipated. put it back in pool and look for best again
                current.remainingTimeEstimated = realRemainingTime;
                posPool.add(current);
            } else {
                currentGood = true;
                visited((int) current.sceneSnapshot.getMarioFloatPos()[0], (int) current.sceneSnapshot.getMarioFloatPos()[1], current.timeElapsed);
                posPool.addAll(current.generateChildren());
            }
            if (currentGood) {
                if (bestPosition.getRemainingTime() > current.getRemainingTime())
                    bestPosition = current;
                if (current.sceneSnapshot.getMarioFloatPos()[0] > furthestPosition.sceneSnapshot.getMarioFloatPos()[0])
                    furthestPosition = current;
            }
        }
        if (current.sceneSnapshot.getMarioFloatPos()[0] - currentSearchStartingMarioXPos < maxRight
                && furthestPosition.sceneSnapshot.getMarioFloatPos()[0] > bestPosition.sceneSnapshot.getMarioFloatPos()[0] + 20)
            // Couldnt plan till end of screen, take furthest
            bestPosition = furthestPosition;

        return current.sceneSnapshot;
    }

    private void startSearch(MarioForwardModel model, int repetitions) {
        SearchNode startPos = new SearchNode(null, repetitions, null);
        startPos.initializeRoot(model);

        posPool = new ArrayList<SearchNode>();
        visitedStates.clear();
        posPool.addAll(startPos.generateChildren());
        currentSearchStartingMarioXPos = model.getMarioFloatPos()[0];

        bestPosition = startPos;
        furthestPosition = startPos;
    }

    private ArrayList<boolean[]> extractPlan() {
        ArrayList<boolean[]> actions = new ArrayList<boolean[]>();

        // just move forward if no best position exists
        if (bestPosition == null) {
            for (int i = 0; i < 10; i++) {
                actions.add(Helper.createAction(false, true, false, false, true));
            }
            return actions;
        }

        SearchNode current = bestPosition;
        while (current.parentPos != null) {
            for (int i = 0; i < current.repetitions; i++)
                actions.add(0, current.action);
            if (current.hasBeenHurt) {
                requireReplanning = true;
            }
            current = current.parentPos;
        }
        return actions;
    }

    private SearchNode pickBestPos(ArrayList<SearchNode> posPool) {
        SearchNode bestPos = null;
        float bestPosCost = 10000000;
        for (SearchNode current : posPool) {
            float currentCost = current.getRemainingTime() + current.timeElapsed * 0.90f; // slightly bias towards furthest positions
            if (currentCost < bestPosCost) {
                bestPos = current;
                bestPosCost = currentCost;
            }
        }
        posPool.remove(bestPos);
        return bestPos;
    }

    public boolean[] optimise(MarioForwardModel model, MarioTimer timer) {
        int planAhead = 2;
        int stepsPerSearch = 2;

        MarioForwardModel originalModel = model.clone();
        ticksBeforeReplanning--;
        requireReplanning = false;
        if (ticksBeforeReplanning <= 0 || currentActionPlan.size() == 0 || requireReplanning) {
            currentActionPlan = extractPlan();
            if (currentActionPlan.size() < planAhead) {
                planAhead = currentActionPlan.size();
            }

            // simulate ahead to predicted future state, and then plan for this future state
            for (int i = 0; i < planAhead; i++) {
                model.advance(currentActionPlan.get(i));
            }
            startSearch(model, stepsPerSearch);
            ticksBeforeReplanning = planAhead;
        }
        if (model.getGameStatus() == GameStatus.LOSE) {
            startSearch(originalModel, stepsPerSearch);
        }
        search(timer);

        boolean[] action = new boolean[5];
        if (currentActionPlan.size() > 0)
            action = currentActionPlan.remove(0);
        return action;
    }

    private void visited(int x, int y, int t) {
        visitedStates.add(new int[]{x, y, t});
    }

    private boolean isInVisited(int x, int y, int t) {
        int timeDiff = 5;
        int xDiff = 2;
        int yDiff = 2;
        for (int[] v : visitedStates) {
            if (Math.abs(v[0] - x) < xDiff && Math.abs(v[1] - y) < yDiff && Math.abs(v[2] - t) < timeDiff
                    && t >= v[2]) {
                return true;
            }
        }
        return false;
    }

}
