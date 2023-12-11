package agents.spencerSchumann;

import java.util.ArrayList;

import engine.core.MarioForwardModel;

/**
 * @author Spencer Schumann
 */
public class Scene {

    public ArrayList<Edge> floors = new ArrayList<Edge>();
    public ArrayList<Edge> walls = new ArrayList<Edge>();
    public ArrayList<Edge> ceilings = new ArrayList<Edge>();
    public ArrayList<BumpableEdge> bumpables = new ArrayList<BumpableEdge>();
    public ArrayList<Edge> enemyEmitters = new ArrayList<Edge>();
    public long constructTime;

    // The extreme top left corner of the scene, in world coordinates
    public float originX;
    public float originY;

    @Override
    public Scene clone() {
        // TODO: I should really be calling s.clone() here; see Object.clone().
        Scene s = new Scene();
        s.update(this);
        return s;
    }

    public void update(Scene scene) {
        // TODO: no deep copy of edges; will this cause problems?
        clearEdges();
        add(scene);
        originX = scene.originX;
        originY = scene.originY;
    }

    public void clearEdges() {
        floors.clear();
        walls.clear();
        ceilings.clear();
        bumpables.clear();
        enemyEmitters.clear();
    }

    private Scene() {
    }

    public Scene(float originX, float originY) {
        this.originX = originX;
        this.originY = originY;
    }

    public Scene(MarioForwardModel model, int[][] scene) {
        long startTime = System.nanoTime();

        float[] marioPos = model.getMarioFloatPos();
        originX = (float) Math.floor(marioPos[0] / 16.0f) * 16.0f - (model.obsGridWidth / 2) * 16.0f;
        originY = (float) Math.floor(marioPos[1] / 16.0f) * 16.0f - (model.obsGridHeight / 2) * 16.0f;

        boolean[][] visited = new boolean[scene.length][scene[0].length];
        int x, y;
        for (x = 0; x < scene.length; x++) {
            for (y = 0; y < scene[x].length; y++) {
                int tile = scene[x][y];
                if (tile == Tiles.COIN) {
                    // TODO
                } else if (!visited[y][x]) {
                    if (Tiles.isWall(tile)) {
                        Scene block = new Scene(originX, originY);
                        block.expandWall(scene, visited, x, y);
                        add(block);
                    } else if (tile == Tiles.JUMPTHROUGH) {
                        Scene ledge = new Scene(originX, originY);
                        ledge.expandLedge(scene, visited, x, y);
                        add(ledge);
                    }
                }
            }
        }
        // TODO: update bumpables and EnemyEmitters, if not already done
        constructTime = System.nanoTime() - startTime;
    }

    // Expand vectorized block from an initial starting point, and mark
    // all tiles that are part of this block as visited
    private void expandWall(int[][] scene, boolean[][] visited, int x, int y) {
        if (visited[y][x]) {
            return;
        }
        visited[y][x] = true;
        // left side
        if (x > 0) {
            if (Tiles.isWall(scene[x - 1][y])) {
                expandWall(scene, visited, x - 1, y);
            } else {
                walls.add(new Edge(originX + x * 16.0f, originY + y * 16.0f,
                        originX + x * 16.0f, originY + (y + 1) * 16.0f));
            }
        }
        // right side
        if (x < scene.length - 1) {
            if (Tiles.isWall(scene[x + 1][y])) {
                expandWall(scene, visited, x + 1, y);
            } else {
                walls.add(new Edge(originX + (x + 1) * 16.0f, originY + y * 16.0f,
                        originX + (x + 1) * 16.0f, originY + (y + 1) * 16.0f));
            }
        }
        // top side
        if (y > 0) {
            if (Tiles.isWall(scene[x][y - 1])) {
                expandWall(scene, visited, x, y - 1);
            } else {
                floors.add(new Edge(originX + x * 16.0f, originY + y * 16.0f,
                        originX + (x + 1) * 16.0f, originY + y * 16.0f));
            }
        }
        // bottom side
        if (y < scene[x].length - 1) {
            if (Tiles.isWall(scene[x][y + 1])) {
                expandWall(scene, visited, x, y + 1);
            } else {
                ceilings.add(new Edge(originX + x * 16.0f, originY + (y + 1) * 16.0f,
                        originX + (x + 1) * 16.0f, originY + (y + 1) * 16.0f));
            }
        }
        coalesce();
    }

    // Expand ledge
    private void expandLedge(int[][] scene, boolean[][] visited, int x, int y) {
        if (visited[y][x]) {
            return;
        }
        visited[y][x] = true;
        int startx = x;
        int endx = x;
        // Find left side of ledge
        while (startx > 0 && scene[startx - 1][y] == Tiles.JUMPTHROUGH) {
            startx--;
            visited[y][startx] = true;
        }
        // Find right side of ledge
        while (endx < scene.length - 1 && scene[endx + 1][y] == Tiles.JUMPTHROUGH) {
            endx++;
            visited[y][endx] = true;
        }
        floors.add(new Edge(originX + startx * 16.0f, originY + y * 16.0f,
                originX + (endx + 1) * 16.0f, originY + y * 16.0f));
    }

    // Coalesce adjacent edges of the same type into one
    private void coalesce() {
        // TODO
        // Note: should I have a contiguous ceiling, or break it up for
        // each special?
        coalesce(walls);
        coalesce(ceilings);
        coalesce(floors);
        coalesce(enemyEmitters);
        // NOTE: bumpables shouldn't be coalesced.
    }

    private void coalesce(ArrayList<Edge> edges) {
        // Super stupid way for now.
        // TODO: optimize.  Without coalesce, everything runs in about
        // 60 microseconds max.  With coalesce, times get up to around 6
        // milliseconds.  That's a large chunk of the allotted 40 ms to
        // be wasting on this.
        boolean foundOne = true;
        while (foundOne) {
            foundOne = false;
            for (Edge a : edges) {
                for (Edge b : edges) {
                    if (a == b) {
                        continue;
                    }
                    foundOne = true;
                    if (a.x1 == b.x1 && a.y1 == b.y1) {
                        // Overlapping?  Something is wrong...
                        throw new RuntimeException("Overlapping edges!");
                    } else if (a.x1 == b.x2 && a.y1 == b.y2) {
                        a.x1 = b.x1;
                        a.y1 = b.y1;
                    } else if (a.x2 == b.x1 && a.y2 == b.y1) {
                        a.x2 = b.x2;
                        a.y2 = b.y2;
                    } else {
                        foundOne = false;
                    }

                    if (foundOne) {
                        edges.remove(b);
                        break;
                    }
                }
                if (foundOne) {
                    break;
                }
            }
        }
    }

    // Add the edges in the subscene to this scene
    private void add(Scene subscene) {
        floors.addAll(subscene.floors);
        walls.addAll(subscene.walls);
        ceilings.addAll(subscene.ceilings);
        bumpables.addAll(subscene.bumpables);
        enemyEmitters.addAll(subscene.enemyEmitters);
    }
}