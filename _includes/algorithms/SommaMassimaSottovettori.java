package divideetimpera;

public class SommaMassimaSottovettori {
	
	private static class Subarray implements Cloneable {
		int start, end, sum;
		public Subarray(int i, int j, int somma) {
			start = i; end = j; sum = somma;
		}
		public Object clone() {
			return new Subarray(start, end, sum);
		}
		public String toString() {
			return String.format("(%d,%d) -> %d", start, end, sum);
		}
	}
	
	private static Subarray calculateMaxSum(int A[], int i, int j) {
		if(i == j) {
			return new Subarray(i, i, A[i]);
		}
		int middle = (i+j)/2;
		
		Subarray left = calculateMaxSum(A, i, middle);
		Subarray right = calculateMaxSum(A, middle+1, j);
		Subarray esteso = new Subarray(left.start, right.end, 0);
		for(int index = left.start; index <= right.end; index++)
			esteso.sum += A[index];
		
		Subarray risultato = esteso;
		if(left.sum > risultato.sum)  risultato = left;
		if(right.sum > risultato.sum) risultato = right;
		return risultato;
	}
	
	public static void main(String args[]) {
		int array[] = new int[]{3, -1, 3, 5, -8, 5};
		System.out.println("Il massimo sottoarray è " 
				+ calculateMaxSum(array, 0, array.length-1));
	}
}
