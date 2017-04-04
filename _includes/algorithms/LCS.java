package programmazionedinamica;

import util.MatrixPrinter;

public class LCS {
	
	private static final int VUOTO = -1;
	
	private final String X;
	private final String Y;
	private final int mat[][];
	
	public LCS(String x, String y) {
		X = x;
		Y = y;
		mat = new int[X.length()][Y.length()];
		for(int i = 0; i < mat.length; i++)
			for(int j = 0; j < mat[i].length; j++)
				mat[i][j] = VUOTO;
	}
	
	private boolean matches(int i, int j) {
		return X.charAt(i) == Y.charAt(j);
	}
	
	private int fillMatrix(int i, int j) {
		if(i < 0 || j < 0) // raggiunti bordi della matrice
			return 0;
		else if(mat[i][j] != VUOTO) // già calcolato
			return mat[i][j];
		
		if(matches(i, j)) {
			mat[i][j] = 1 + fillMatrix(i - 1, j - 1);
			
		} else {
			int up = fillMatrix(i - 1, j);
			int left = fillMatrix(i, j - 1);
			mat[i][j] = Math.max(up, left);
		}
		
		return mat[i][j];
	}
	
	public void fillMatrix() {
		fillMatrix(X.length()-1, Y.length()-1);
	}
	
	public String printSolution() {
		StringBuilder solution = new StringBuilder();
		
		int i = X.length()-1, j = Y.length()-1;
		while(i >= 0 && j >= 0) {
			if(matches(i, j)) {
				solution.append(X.charAt(i));
				i--;
				j--;
			} else if(i < 1) { // non posso andare a sinistra
				j--;
			} else if(j < 1) { // non posso andare in alto
				i--;
			} else { // posso andare dappertutto
				if(mat[i-1][j] > mat[i][j-1]) { // left > up
					i--;
				} else { // up > left
					j--;
				}
			}
		}
		
		return solution.reverse().toString();
	}
	
	public static void main(String args[]) {
		LCS lcs = new LCS("TORTA", "ORTAO");
		lcs.fillMatrix();
		MatrixPrinter.print(lcs.mat, "X", "Y");
		System.out.println("La soluzione e': " + lcs.printSolution());
	}
}