package backtracking;

public abstract class Backtrackable<Data> {

	protected boolean finished = false;
	
	protected void backtrack(int a[], int k, Data input) {
		if(is_a_solution(a, k, input)) {
			process_solution(a, k, input);
		} else {
			k = k + 1;
			int c[] = construct_candidates(a, k, input);
			for(int i = 0; i < c.length; i++) {
				a[k] = c[i];
				make_move(a, k, input);
				backtrack(a, k, input);
				unmake_move(a, k, input);
				if(finished) return;
			}
		}
	}
	
	// test whether the first k elements completes a solution for the
	// problem. input allowed to pass general information into the
	// routine (eg n = size of target solution)
	protected abstract boolean is_a_solution(int a[], int k, Data input);
	
	// fills and return an array with the complete set of possible candidates
	// for the k-th position of a, given the contents of the first k-1 positions
	protected abstract int[] construct_candidates(int a[], int k, Data input);
	
	// process the current solution
	protected abstract void process_solution(int a[], int k, Data input);
	
	// modify a data structure in response to the latest move and
	// clean up if we decide to take back a move
	protected abstract void make_move(int a[], int k, Data input);
	protected abstract void unmake_move(int a[], int k, Data input);
}
