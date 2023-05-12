package agents.andySloane;

import java.util.*;

public class PrioQ {
    private static MarioStateComparator comparator = new MarioStateComparator();
    private MarioState[] queue;
    private int size = 0;

    public PrioQ(int initialCapacity) {
        this.queue = new MarioState[initialCapacity];
    }

    public boolean offer(MarioState e) {
        if (e == null)
            throw new NullPointerException();
        if (size == queue.length)
            drop();
        int i = size;
        size = i + 1;
        if (i == 0)
            queue[0] = e;
        else
            siftUp(i, e);
        return true;
    }

    public boolean isEmpty() {
        return size == 0;
    }

    public int size() {
        return size;
    }

    public void clear() {
        for (int i = 0; i < size; i++)
            queue[i] = null;
        size = 0;
    }

    public MarioState peek() {
        if (size == 0)
            return null;
        return queue[0];
    }

    public MarioState poll() {
        if (size == 0)
            return null;
        int s = --size;
        MarioState result = queue[0];
        MarioState x = queue[s];
        queue[s] = null;
        if (s != 0)
            siftDown(0, x);
        return result;
    }

    public void drop() {
        Arrays.sort(queue, 0, size, comparator);
        int l = size >> 1;
        for (int i = 0; i < l; ++i)
            queue[--size] = null;
        for (int i = (size >>> 1) - 1; i >= 0; i--)
            siftDown(i, queue[i]);
    }

    private void siftUp(int k, MarioState x) {
        while (k > 0) {
            int parent = (k - 1) >>> 1;
            MarioState e = queue[parent];
            if (compare(x, e) >= 0)
                break;
            queue[k] = e;
            k = parent;
        }
        queue[k] = x;
    }

    private void siftDown(int k, MarioState x) {
        int half = size >>> 1;
        while (k < half) {
            int child = (k << 1) + 1;
            MarioState c = queue[child];
            int right = child + 1;
            if (right < size && compare(c, queue[right]) > 0)
                c = queue[child = right];
            if (compare(x, c) <= 0)
                break;
            queue[k] = c;
            k = child;
        }
        queue[k] = x;
    }

    private int compare(MarioState a, MarioState b) {
        if (a.cost < b.cost)
            return -1;
        if (a.cost > b.cost)
            return 1;
        return 0;
    }
}
