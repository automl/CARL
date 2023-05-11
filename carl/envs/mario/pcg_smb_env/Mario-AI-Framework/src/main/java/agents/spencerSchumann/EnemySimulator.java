/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package agents.spencerSchumann;

import java.util.ArrayList;

import engine.core.MarioForwardModel;

/**
 * @author Spencer
 */
public class EnemySimulator {

    public ArrayList<Enemy> enemies;

    public EnemySimulator() {
        enemies = new ArrayList<Enemy>();
    }

    @Override
    public EnemySimulator clone() {
        EnemySimulator es = new EnemySimulator();
        for (Enemy e : enemies) {
            es.enemies.add(e.clone());
        }
        return es;
    }

    // Update all known enemy positions based on given scene
    public void update(Scene scene) {
        for (Enemy enemy : enemies) {
            // TODO: this needs to be much more sophisticated.
            enemy.x += enemy.vx;
        }
    }

    // Add new enemies based on the given observation.
    // update(Scene scene) should be called first.
    public void update(MarioForwardModel model) {
        float[] ep = model.getEnemiesFloatPos();
        int i;
        ArrayList<Enemy> newEnemies = new ArrayList<Enemy>();
        for (i = 0; i < ep.length; i += 3) {
            int type = (int) ep[i];
            float x = ep[i + 1];
            float y = ep[i + 2];
            boolean found = false;
            for (Enemy enemy : enemies) {
                if (type == enemy.type && y == enemy.y) {
                    float xdiff = Math.abs(enemy.x - x);
                    if (xdiff == 0.0f) {
                        // Enemy in new observation was already known.
                        newEnemies.add(enemy);
                        enemies.remove(enemy);
                        found = true;
                        break;
                    } else if (xdiff < 5.0f) {
                        enemy.vx += x - enemy.x;
                        enemy.x = x;
                        newEnemies.add(enemy);
                        enemies.remove(enemy);
                        found = true;
                        break;
                    }
                }
            }
            if (!found) {
                // Not found; create new enemy
                newEnemies.add(new Enemy(type, x, y));
            }
        }
        enemies = newEnemies;
    }
}
