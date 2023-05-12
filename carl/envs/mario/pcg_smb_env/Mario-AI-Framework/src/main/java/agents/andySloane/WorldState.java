package agents.andySloane;

import java.util.HashMap;
import java.util.Vector;

public final class WorldState {
    public int[][] map;
    public int[] heightmap;
    public int MapX, MapY;

    // List of currently known enemies; maintained sorted by x coordinate
    public Vector<SpriteState> enemies, addqueue;

    WorldState pred = null;
    HashMap<WSHashKey, WorldState> succ; // successor map

    // hash key comparator
    private class WSHashKey {
        static final int MOD_NONE = 0;
        static final int MOD_REMOVETILE = 1;
        static final int MOD_STOMP = 2;
        public int modType = MOD_NONE;
        public int modTile = 0;
        public SpriteState modEnemy = null;

        WSHashKey() {
            modType = MOD_NONE;
        }

        WSHashKey(int _modTile) {
            modType = MOD_REMOVETILE;
            modTile = _modTile;
        }

        @Override
        public int hashCode() {
            switch (modType) {
                case MOD_NONE:
                    return 0;
                case MOD_REMOVETILE:
                    return 1 + modTile * 4;
                case MOD_STOMP:
                    return 2 + modEnemy.hashCode() * 4;
            }
            return -1;
        }

        @Override
        public boolean equals(Object _o) {
            WSHashKey o = (WSHashKey) _o;
            if (o.modType != modType)
                return false;
            if (o.modTile != modTile)
                return false;
            return o.modEnemy == modEnemy;
        }
    }

    public WorldState(int[][] _map, MarioState ms, float[] enemyPosition) {
        map = _map;
        MapX = (int) ms.x / 16 - 8;
        MapY = (int) ms.y / 16 - 8;
        succ = new HashMap<WSHashKey, WorldState>();
        enemies = new Vector<SpriteState>();
        buildHeightMap();
        syncEnemies(this, enemyPosition, ms);
    }

    WorldState() {
    }

    public WorldState clone() {
        WorldState w = new WorldState();
        w.map = map;
        w.MapX = MapX;
        w.MapY = MapY;
        w.heightmap = heightmap;
        w.succ = new HashMap<WSHashKey, WorldState>();
        w.enemies = enemies; // share enemies vector by default
        w.addqueue = addqueue;
        return w;
    }

    // nondestructive step
    @SuppressWarnings("unchecked")
    public WorldState step() {
        WSHashKey h = new WSHashKey();
        WorldState s = succ.get(h);
        if (s == null) {
            s = clone();
            s.enemies = (Vector<SpriteState>) enemies.clone();
            s.stepEnemies();
            succ.put(h, s);
        }
        return s;
    }

    // destructive update, but returns new worldstate. bleh, it's a mess.
    public void sync(WorldState prevws, int[][] _map, MarioState ms, float[] enemyPosition) {
        map = _map;
        MapX = (int) ms.x / 16 - 8;
        MapY = (int) ms.y / 16 - 8;
        buildHeightMap();
        succ.clear();
        syncEnemies(prevws, enemyPosition, ms);
    }

    void buildHeightMap() {
        heightmap = new int[16];
        // System.out.printf("heightmap: ");
        for (int i = 0; i < 16; i++) {
            int j;
            for (j = 15; j >= 0; j--) // find the first block from the bottom
                if (map[j][i] != 0)
                    break;
            if (j < 0 || j + MapY < 8) { // this is probably a ceiling
                heightmap[i] = 22;
            } else {
                for (; j >= 0; j--)
                    if (map[j][i] == 0 || map[j][i] == 1)
                        break; // 1 is mario, 0 is blank
                heightmap[i] = j + 1;
            }
            // System.out.printf("%02d ", heightmap[i]);
        }
        // System.out.printf("\n");
    }

    //////////////////////////////////////////////
    // destructive operations
    void _removeTile(WSHashKey h, int x, int y) {
        int[][] newmap = new int[16][16];
        for (int j = 0; j < 16; j++)
            for (int i = 0; i < 16; i++)
                newmap[i][j] = map[i][j];
        newmap[x][y] = 0;
        map = newmap;
    }

    private static class EnemyObservation implements Comparable<EnemyObservation> {
        int type;
        float x, y;

        EnemyObservation(int type, float x, float y) {
            this.type = type;
            this.x = x;
            this.y = y;
        }

        public int compareTo(EnemyObservation b) {
            return x < b.x ? -1 : x > b.x ? 1 : 0;
        }
    }

    // this function is terrible and slow, but it only needs to be done once per
    // real frame.
    public void syncEnemies(WorldState prevws, float[] enemyObs, MarioState ms) {
        // when we get a new observation, sort the observation by x and filter
        // through the list, using the nearest enemy of the same type and comparing
        // predicted states with actual
        EnemyObservation[] obs = new EnemyObservation[enemyObs.length / 3];
        for (int i = 0; i < enemyObs.length; i += 3)
            obs[i / 3] = new EnemyObservation((int) enemyObs[i], enemyObs[i + 1], enemyObs[i + 2]);

        Vector<SpriteState> newenemies = new Vector<SpriteState>(enemies.size() + 2);
        Vector<SpriteState> oldenemies = prevws.enemies;

        // merge enemy observations into our internal enemy array
        for (EnemyObservation eobs : obs) {
            SpriteState closest = null;
            float closestdist = Float.POSITIVE_INFINITY;
            int closest_idx = 0;
            for (int i = 0; i < enemies.size(); i++) {
                SpriteState s = enemies.get(i);
                if (s.type != eobs.type)
                    continue;
                float ex = s.x - eobs.x;
                float ey = s.y - eobs.y;
                float dist = ex * ex + ey * ey;
                if (closest == null || dist < closestdist) {
                    closest = s;
                    closestdist = dist;
                    closest_idx = i;
                }
            }
            if (closest == null || closestdist > 64) { // allow a slop of 8 pixels
                // assume new enemy
                closest = SpriteState.newEnemy(eobs.x, eobs.y, eobs.type, ms);
            } else {
                if (closestdist != 0) {
                    if (closest_idx >= oldenemies.size()) {
                        // if this was newly created but incorrectly i guess we
                        // have to force a recreation
                        closest = SpriteState.newEnemy(eobs.x, eobs.y, eobs.type, ms);
                    } else {
                        SpriteState prev = oldenemies.get(closest_idx);
                        closest.resync(eobs.x, eobs.y, prev.x, prev.y);
                    }
                }
            }
            if (closest != null)
                newenemies.add(closest);
        }
        enemies = newenemies;
    }

    public void stepEnemies() {
        for (int i = 0; i < enemies.size(); i++) {
            SpriteState e = enemies.get(i).clone();
            boolean keep = e.move(this);
            if (keep) {
                enemies.set(i, e);
            } else {
                enemies.remove(i);
                i--;
            }
        }
    }

    // interact with mario after everyone does their move step
    // destructively updates MarioState, but non-destructively returns updated
    // WorldState
    public WorldState interact(MarioState ms, boolean verbose) {
        WorldState ws = this;
        ws.addqueue = new Vector<SpriteState>();
        int i;
        if (verbose)
            System.out.printf("--interact\n");
        for (i = 0; i < ws.enemies.size(); i++) {
            ws = ws.enemies.get(i).collideCheck(ws, ms);
        }
        // now bring in the added stuff
        for (SpriteState s : ws.addqueue) {
            if (verbose) {
                System.out.printf("interact: new e t=%d xy=%f,%f xaya=%f,%f deadTime=%d\n", s.type, s.x, s.y, s.xa,
                        s.ya, s.deadTime);
            }
            ws.enemies.add(s);
        }
        ws.addqueue = null;
        return ws;
    }

    public void addShell(float x, float y) {
        ShellState s = new ShellState(x, y, true);
        s.move(this);
        addqueue.add(s);
    }

    //////////////////////////////////////////////
    // functional operations
    WorldState removeTile(int x, int y) {
        x -= MapX;
        y -= MapY;
        if (x < 0 || x >= 16 || y < 0 || y >= 16)
            return this;

        WSHashKey h = new WSHashKey(x * 16 + y);
        WorldState s = succ.get(h);
        if (s == null) {
            s = clone();
            s._removeTile(h, x, y);
            succ.put(h, s);
        }
        return s;
    }

    final int getBlock(int x, int y) {
        // move x,y world coordinates to the 22x22 reference frame
        x -= MapX;
        y -= MapY;
        if (x < 0 || x >= 16 || y < 0 || y >= 16)
            return 0;

        return map[x][y];
    }

    final boolean isBlocking(int x, int y, float xa, float ya) {
        int block = getBlock(x, y);

        if (block == 1)
            return false; // mario; ignore
        if (block == 34)
            return false; // coin
        if (block == -11)
            return ya > 0; // platform
        return block != 0;
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    final WorldState stomp(SpriteState e, MarioState ms) {
        // destructively modify mario
        ms.stomp(e);
        // clone us, and clone e, and splice e in the array
        WorldState ws = clone();
        ws.enemies = (Vector) enemies.clone();
        ws.enemies.set(ws.enemies.indexOf(e), e.stomp(this, ms));
        return ws;
    }

    final WorldState bump(int x, int y, boolean big) {
        // System.out.printf("bumping tile @%d,%d = %d\n", x,y,getBlock(x,y));
        if (big) {
            switch (getBlock(x, y)) {
                case 16: // regular brick
                    // unfortunately it could also be a hidden coin or
                    // something else
                    return removeTile(x, y);
            }
        }
        return this;
    }

    // this is destructive, done during interact(), unlike bump and stomp
    // (which are mario-initiated actions)
    final void checkShellCollide(ShellState s) {

    }

    final void kick(ShellState s) {
    }
}
