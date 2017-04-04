package backtracking;

import java.util.*;

public class ConstructAllMultiset 
		extends Backtrackable<List<Integer>> {

	@Override
	protected boolean is_a_solution
			(int[] a, int k, List<Integer> remaining) {
		
		return remaining.size() == 0;
	}

	@Override
	protected int[] construct_candidates
			(int[] a, int k, List<Integer> remaining) {
		
		Set<Integer> candidates = new TreeSet<>(remaining);
		int c[] = new int[candidates.size()], i = 0;
		for(Integer candidate : candidates)
			c[i++] = candidate;
		
		return c;
	}

	@Override
	protected void process_solution
			(int[] a, int k, List<Integer> remaining) {
		
		System.out.println(Arrays.toString(a));
	}

	@Override
	protected void make_move
		(int[] a, int k, List<Integer> remaining) {
		
		int current = a[k];
		remaining.remove(new Integer(current));
	}

	@Override
	protected void unmake_move
		(int[] a, int k, List<Integer> remaining) {
		
		int current = a[k];
		remaining.add(new Integer(current));
		
		for(int i = k+1; i < a.length; i++)
			a[i] = 0;
	}
	
	public void start(int elements[]) {
		List<Integer> remaining = new LinkedList<>();
		for(int elem : elements)
			remaining.add(elem);
		
		backtrack(new int[elements.length+1], 0, remaining);
	}

	public static void main(String args[]) {
		int elements[] = new int[]{1, 1, 2, 2, 3};
		(new ConstructAllMultiset()).start(elements);
	}
}
