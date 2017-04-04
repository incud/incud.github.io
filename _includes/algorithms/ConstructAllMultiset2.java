package backtracking;

import java.util.*;
import com.google.common.collect.*;

public class ConstructAllMultiset2 
		extends Backtrackable<Multiset<Integer>> {

	@Override
	protected boolean is_a_solution
			(int[] a, int k, Multiset<Integer> input) {

		return input.size() == 0;
	}

	@Override
	protected int[] construct_candidates
			(int[] a, int k, Multiset<Integer> input) {
		
		// prendo tutti gli elementi RIMANENTI distinti
		Set<Integer> set = input.elementSet();
		// li metto in un array
		int c[] = new int[set.size()], i = 0;
		for(Integer elem : set)
			c[i++] = elem;
		return c;
	}

	@Override
	protected void process_solution
			(int[] a, int k, Multiset<Integer> input) {
		
		int arr[] = Arrays.copyOfRange(a, 1, a.length);
		System.out.println(Arrays.toString(arr));
	}

	@Override
	protected void make_move
			(int[] a, int k, Multiset<Integer> input) {
		
		int current = a[k];
		input.remove(current);
	}

	@Override
	protected void unmake_move
			(int[] a, int k, Multiset<Integer> input) {
		
		int current = a[k];
		input.add(current);
	}
	
	public void start(int elements[]) {
		Multiset<Integer> mset = TreeMultiset.create();
		for(int elem : elements)
			mset.add(elem);
		
		backtrack(new int[elements.length+1], 0, mset);
	}
	
	public static void main(String args[]) {
		int combinations[] = new int[]{1,1,2,2};
		(new ConstructAllMultiset2()).start(combinations);
	}
}
