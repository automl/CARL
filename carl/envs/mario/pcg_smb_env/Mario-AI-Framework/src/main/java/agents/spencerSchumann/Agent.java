package agents.spencerSchumann;

import engine.core.MarioAgent;
import engine.core.MarioForwardModel;
import engine.core.MarioTimer;
import engine.helper.MarioActions;

/**
 * @author Spencer Schumann
 */
public class Agent implements MarioAgent {

    private Tiles tiles;
    private MarioState mario;
    private EnemySimulator enemySim;
    private boolean manualOverride = false;
    private PlanRunner planRunner;

    @Override
    public void initialize(MarioForwardModel model, MarioTimer timer) {
        tiles = new Tiles();
        mario = new MarioState();
        planRunner = new PlanRunner();
        enemySim = new EnemySimulator();
    }

    @Override
    public boolean[] getActions(MarioForwardModel model, MarioTimer timer) {
        float[] marioPos = model.getMarioFloatPos();
        tiles.addObservation(model);
        int mx = (int) (marioPos[0] / 16.0f);
        int my = (int) (marioPos[1] / 16.0f);
        // TODO: in the latest code drop, it looks like there is no "mario hole."
        // So, tiles could be removed.
        int[][] scene = tiles.getScene(mx - model.obsGridWidth / 2, my - model.obsGridHeight / 2,
                model.obsGridWidth, model.obsGridHeight);
        mario.update(model);
        Scene sanitizedScene = new Scene(model, scene);
        enemySim.update(sanitizedScene);
        enemySim.update(model);

        boolean[] action = null;
        if (planRunner.isDone() || planRunner.isLastAction() || manualOverride) {
            MovementPlanner planner = new MovementPlanner(sanitizedScene, mario, enemySim.clone());
            PlanRunner plan = planner.planMovement();
            if (plan != null) {
                //System.out.println("New plan.");
                planRunner = plan;
            } else {
                action = new boolean[5];
                action[MarioActions.RIGHT.getValue()] = true;
                action[MarioActions.SPEED.getValue()] = action[MarioActions.JUMP.getValue()] =
                        model.mayMarioJump() || !model.isMarioOnGround();
            }
        }
        if (action == null)
            action = planRunner.nextAction();
        return action;
    }

    @Override
    public String getAgentName() {
        return "SpencerShumannAgent";
    }

}
