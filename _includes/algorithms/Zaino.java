package programmazionedinamica;

import util.MatrixPrinter;

public class Zaino {
	
	static final int VUOTO = -1;
	final int W;
	final int pesi[], valori[];
	final int mat[][];
	
	public Zaino(int w, int weights[], int values[]) {
		if(weights.length != values.length) 
			throw new IllegalArgumentException();
		W = w; 
		pesi = weights; 
		valori = values;
		mat = new int[valori.length][getSumWeigths() + 1];
		initMatrix();
	}
	
	private int getSumWeigths() {
		int sumW = 0; 
		for(int peso : pesi) 
			sumW += peso;
		return sumW;
	}
	
	private void initMatrix() {
		for(int i = 0; i < mat.length; i++) 
			for(int j = 0; j < mat[i].length; j++)
				mat[i][j] = VUOTO;
	}
	
	// riempie la matrice e ritorna il valore della soluzione (i, w)
	public int riempiMatrice(int i, int w) {
		if(i < 0 || w <= 0) { // casi base
			return 0;
		} else if(mat[i][w] != VUOTO) { // soluzione cachata
			return mat[i][w];
		}
		
		if(w >= pesi[i]) { // scelgo il maggiore tra le opzioni
			int elCorrScelto = valori[i] + riempiMatrice(i - 1, w - pesi[i]);
			int elCorrIgnorato = riempiMatrice(i - 1, w);
			mat[i][w] = Math.max(elCorrScelto, elCorrIgnorato);
		} else { // oggetto corrente non sta nello zaino
			mat[i][w] = riempiMatrice(i - 1, w);
		}
		return mat[i][w];
	}
	
	public int riempiMatrice() { // riempie la matrice e restituisce il val max
		return riempiMatrice(pesi.length-1, W);
	}
	
	private void stampaElemento(int i, boolean scelto) {
		System.out.printf("Elemento %2d %10s (w = %3d, value = %3d)\n", i, scelto ? "scelto" : "ignorato", pesi[i], valori[i]);
	}
	
	private void stampaSoluzione(int i, int w) {
		if(i == 0) { // processo l'ultimo elemento
			stampaElemento(0, w >= pesi[0]);
			return;
		}
		if(w > 0) { // tutti gli elementi, sse w > 0
			if(mat[i][w] == mat[i-1][w]) { // elemento ignorato
				stampaSoluzione(i-1, w);
				stampaElemento(i, false);
			} else { // elemento scelto
				stampaSoluzione(i-1, w - pesi[i]);
				stampaElemento(i, true);
			}
		}
	}
	
	public void stampaSoluzione() { // riempie la matrice e restituisce il val max
		stampaSoluzione(pesi.length-1, W);
	}

	public static void main(String args[]) {
		Zaino zaino = new Zaino(10, 
				new int[]{1, 1, 2, 3, 2, 1, 4, 3, 4, 2, 1, 1}, 
				new int[]{2, 3, 4, 3, 4, 1, 5, 3, 2, 3, 2, 1});
		System.out.println("Il valore massimo che sta nello zaino e': " + zaino.riempiMatrice());
		MatrixPrinter.print(zaino.mat, "i", "W");
		zaino.stampaSoluzione();
	}	
}
