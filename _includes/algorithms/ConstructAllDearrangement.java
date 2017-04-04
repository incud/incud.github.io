package backtracking;

import java.util.*;

public class ConstructAllDearrangement 
		extends Backtrackable<Object> {

	@Override
	protected boolean is_a_solution
			(int[] a, int k, Object niente) {
		
		return k == a.length-1;
	}

	@Override
	protected int[] construct_candidates
			(int[] a, int k, Object niente) {	
		
		TreeSet<Integer> candidates = new TreeSet<>();
		for(int i = 1; i < a.length; i++)
			if(i != k) // not current position
				candidates.add(i);
		for(int i = 1; i <= k; i++) {
			candidates.remove(a[i]);
		}
		
		int c[] = new int[candidates.size()], i = 0;
		for(int cand : candidates)
			c[i++] = cand;
		
		return c;
	}

	@Override
	protected void process_solution
			(int[] a, int k, Object niente) {
		
		System.out.println("Soluzione: " + 
			Arrays.toString(Arrays.copyOfRange(a, 1, a.length)));
	}

	@Override
	protected void make_move(int[] a, int k, Object niente) { }

	@Override
	protected void unmake_move(int[] a, int k, Object niente) { 
		for(int i = k; i < a.length; i++)
			a[i] = 0;
	}
	
	public void start(int n) {
		backtrack(new int[n+1], 0, n);
	}
	
	public static void main(String args[]) {
		(new ConstructAllDearrangement()).start(3);
	}
}
