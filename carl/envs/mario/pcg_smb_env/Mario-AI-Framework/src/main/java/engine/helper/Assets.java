package engine.helper;

import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.GraphicsConfiguration;
import java.awt.Image;
import java.awt.Transparency;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;

import javax.imageio.ImageIO;

public class Assets {
    public static Image[][] mario;
    public static Image[][] smallMario;
    public static Image[][] fireMario;
    public static Image[][] enemies;
    public static Image[][] items;
    public static Image[][] level;
    public static Image[][] particles;
    public static Image[][] font;
    public static Image[][] map;

    public static void init(GraphicsConfiguration gc) {
        try {
            mario = cutImage(gc, "img/mariosheet.png", 32, 32);
            smallMario = cutImage(gc, "img/smallmariosheet.png", 16, 16);
            fireMario = cutImage(gc, "img/firemariosheet.png", 32, 32);
            enemies = cutImage(gc, "img/enemysheet.png", 16, 32);
            items = cutImage(gc, "img/itemsheet.png", 16, 16);
            level = cutImage(gc, "img/mapsheet.png", 16, 16);
            particles = cutImage(gc, "img/particlesheet.png", 16, 16);
            font = cutImage(gc, "img/font.gif", 8, 8);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private static Image getImage(GraphicsConfiguration gc, String imageName) throws IOException {
        BufferedImage source = null;
        try {
            InputStream stream = ClassLoader.getSystemResourceAsStream(imageName);
            source = ImageIO.read(stream);
        } catch (Exception e) {
            e.printStackTrace();
        }
        Image image = gc.createCompatibleImage(source.getWidth(), source.getHeight(), Transparency.BITMASK);
        Graphics2D g = (Graphics2D) image.getGraphics();
        g.setComposite(AlphaComposite.Src);
        g.drawImage(source, 0, 0, null);
        g.dispose();
        return image;
    }

    private static Image[][] cutImage(GraphicsConfiguration gc, String imageName, int xSize, int ySize)
            throws IOException {
        Image source = getImage(gc, imageName);
        Image[][] images = new Image[source.getWidth(null) / xSize][source.getHeight(null) / ySize];
        for (int x = 0; x < source.getWidth(null) / xSize; x++) {
            for (int y = 0; y < source.getHeight(null) / ySize; y++) {
                Image image = gc.createCompatibleImage(xSize, ySize, Transparency.BITMASK);
                Graphics2D g = (Graphics2D) image.getGraphics();
                g.setComposite(AlphaComposite.Src);
                g.drawImage(source, -x * xSize, -y * ySize, null);
                g.dispose();
                images[x][y] = image;
            }
        }

        return images;
    }

}
