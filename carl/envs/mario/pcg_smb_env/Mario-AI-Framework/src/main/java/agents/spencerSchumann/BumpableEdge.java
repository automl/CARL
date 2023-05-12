package agents.spencerSchumann;

/**
 * @author Spencer Schumann
 */
public class BumpableEdge extends Edge {
    // Type is BRICK, SPECIAL_COIN, or SPECIAL_QUESTION

    int type;

    public BumpableEdge(float x1, float y1, float x2, float y2, int type) {
        super(x1, y1, x2, y2);
        this.type = type;
    }
}
