package util;

public class MatrixPrinter {

	public static void print(int mat[][], String rows, String cols) {
		// print header
		String header = rows + "\\" + cols;
		System.out.printf("%4s  ", header.substring(0, Math.min(header.length(), 4)));
		for(int i = 0; i < mat[0].length; i++)
			System.out.printf("%3d ", i);
		// print line
		System.out.print("\n   ---");
		for(int i = 0; i < mat[0].length; i++)
			System.out.print("----");
		// print body
		for(int i = 0; i < mat.length; i++) {
			System.out.printf("\n%3d|  ", i);
			for(int j = 0; j < mat[i].length; j++) {
				System.out.printf("%3d ", mat[i][j]);
			}
		}
		System.out.println();
	}

}
