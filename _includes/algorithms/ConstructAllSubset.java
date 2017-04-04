package backtracking;

public class ConstructAllSubset extends Backtrackable<Object> {

	@Override
	protected boolean is_a_solution
			(int[] a, int k, Object niente) {
		
		return k == a.length-1;
	}

	@Override
	protected int[] construct_candidates
			(int[] a, int k, Object niente) {
		
		return new int[]{0, 1}; // {true, false}
	}

	@Override
	protected void process_solution
			(int[] a, int k, Object input) {
		
		String solution = "";
		for(int i = 1; i <= k; i++)
			if(a[i] == 1)
				solution += i + " ";
		System.out.printf("{%s}\n", solution);
	}

	@Override
	protected void make_move(int[] a, int k, Object niente) { }

	@Override
	protected void unmake_move(int[] a, int k, Object niente) { }
	
	public void start(int n) {
		backtrack(new int[n+1], 0, n);
	}

	public static void main(String args[]) {
		ConstructAllSubset cas = new ConstructAllSubset();
		cas.start(3);
	}
}