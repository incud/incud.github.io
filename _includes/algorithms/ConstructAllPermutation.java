package backtracking;

import java.util.Arrays;

public class ConstructAllPermutation 
		extends Backtrackable<Object> {

	@Override
	protected int[] construct_candidates
			(int[] a, int k, Object niente) {
		
		boolean in_perm[] = new boolean[a.length];
		Arrays.fill(in_perm, false);
		
		for(int i = 0; i < k; i++)
			in_perm[a[i]] = true;
		
		int c[] = new int[a.length-1];
		int candidates = 0;
		for(int i = 1; i < a.length; i++)
			if(in_perm[i] == false) {
				c[candidates] = i;
				candidates++;
			}

		return Arrays.copyOfRange(c, 0, candidates);
	}
	

	@Override
	protected boolean is_a_solution
			(int[] a, int k, Object niente) {
		
		return k == a.length-1;
	}

	@Override
	protected void process_solution
			(int[] a, int k, Object niente) {
		
		for(int i = 1; i <= k; i++)
			System.out.printf(" %d", a[i]);
		System.out.println();
	}

	@Override
	protected void make_move(int[] a, int k, Object niente) { }

	@Override
	protected void unmake_move(int[] a, int k, Object niente) { }
	
	public void start(int n) { 
		backtrack(new int[n+1], 0, null);
	}

	public static void main(String args[]) {
		(new ConstructAllPermutation()).start(4);
	}
}
