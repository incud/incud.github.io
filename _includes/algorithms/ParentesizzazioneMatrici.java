package programmazionedinamica;

import util.MatrixPrinter;

public class ParentesizzazioneMatrici {
	
	private final int N;
	private final int p[];
	private final int m[][], s[][];
	
	public ParentesizzazioneMatrici(int dim[]) {
		p = dim;
		N = p.length-1;
		m = new int[N][N];
		s = new int[N][N];
	}
	
	public int p(int i) {
		return p[i+1];
	}
	
	public void calculateCost() {
		for(int n = 0; n < N; n++)
			for(int i = 0; i < N-n; i++)
				calculateCell(i, i+n);
	}
	
	private void calculateCell(int i, int j) {
		if(i == j) { // caso base
			m[i][i] = 0;
			s[i][i] = -1;
		} else { // passo ricorsivo
			m[i][j] = Integer.MAX_VALUE;
			for(int k = i; k <= j-1; k++) {
				int cost = m[i][k] + m[k+1][j] + p(i-1)*p(k)*p(j);
				if(cost < m[i][j]) {
					m[i][j] = cost;
					s[i][j] = k;
				}
			}
		}
	}
	
	private String getParenthesis(int i, int j) {
		if(i == j) {
			return String.format("A%d", i+1);
		} else {
			int k = s[i][j];
			return String.format("(%s %s)", 
					getParenthesis(i, k), 
					getParenthesis(k+1, j));
		}
	}
	
	public String toString() {
		return getParenthesis(0, N-1);
	}
	
	public static void main(String[] args) {
		int costi[] = new int[]{7, 8, 4, 2, 3, 5, 6};
		ParentesizzazioneMatrici pm = new ParentesizzazioneMatrici(costi);
		pm.calculateCost();
		MatrixPrinter.print(pm.m, "i", "j");
		System.out.println();
		MatrixPrinter.print(pm.s, "i", "j");
		System.out.println(pm.toString());
	}
}
