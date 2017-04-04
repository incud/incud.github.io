package backtracking;

import java.util.*;
import util.Trie;
import com.google.common.collect.*;

public class Anagrammi {
	
	// ================ DIZIONARIO ================ 
	private static final Trie dizionario;
	static {
		dizionario = new Trie();
		dizionario.addAll(new String[]{
			"ago", "baci", "bacino", "bagno", "bianco", 
			"bigami", "bigamo", "boa", "cambio", "ci", 
			"cibi", "cibo", "cigno", "cingo", "cio", 
			"combini", "combino", "coni", "conio", 
			"cono", "gambi", "gambo", "gia", "gnomi", 
			"gnomo", "in", "incombi", "incombo", "io", 
			"ma", "mago", "magoni", "mangio", "mi", 
			"micio", "mio", "mogi", "no", "noci", "noi", 
			"ogni"});
	}
	
	// ================ PROGRAMMA ================ 
	Multiset<Character> remaining;
	List<String> soluzioni;
	
	private Multiset<Character> build_multiset
			(String phrase) {
		
		Multiset<Character> mset 
			= ConcurrentHashMultiset.create();
		phrase = phrase.toLowerCase();
		for(int i = 0; i < phrase.length(); i++) {
			char c = phrase.charAt(i);
			if(Character.isAlphabetic(c)) {
				mset.add(c);
			}
		}
		return mset;
	}
	
	private String build_word 
			(char[] processed, int startAtIndex, int endAtIndex) {
		
		return String.valueOf(
				processed, startAtIndex, endAtIndex-startAtIndex);
	}
	
	private boolean is_a_word
			(char [] processed, int startAtIndex, int endAtIndex) {
		
		return dizionario.contains(
				build_word(processed, startAtIndex, endAtIndex));
	}
	
	private String indent(int n) { 
	 	return n == 0 ? "" : "\t" + indent(n-1); 
	}
	
	private void backtrack(
			char[] processed, 
			int current, int startAtIndex, int n, 
			String phrase) {
		
		// trovata soluzione 
		if(current == n && 
				is_a_word(processed, startAtIndex, current)) {
			
			// stampo la soluzione corrente 
			String lastWord = build_word(processed, startAtIndex, n);
			soluzioni.add(phrase + " " + lastWord);
		} // cerco le altre soluzioni 
		else {
			//System.out.printf("%sProcesso current = %d, processed = '%s'\n", 
			//		indent(current), current, build_word(processed, 0, current));
			Set<Character> candidates = remaining.elementSet();
			//System.out.printf("%sCandidati: %s\n", 
			//		indent(current), candidates.toString());
			
			for(Character c : candidates) {
				// aggiungo il carattere
				processed[current++] = c;
				remaining.remove(c);
				
				if(is_a_word(processed, startAtIndex, current)) {
					
					// considero tutto
					backtrack(processed, current, startAtIndex, 
							n, phrase);
					
					// considero la rimanenza dei caratteri
					String lastWord = build_word(
							processed, startAtIndex, current);
					
					backtrack(processed, current, current, 
							n, phrase + " " + lastWord);
					
				} else if(dizionario.startsWith(
						build_word(processed, startAtIndex, current))) {
					
					backtrack(processed, current, startAtIndex, 
							n, phrase);
					
				} else {
					//System.out.printf("%sTaglio a current = %d, processed = %s\n", 
					//		indent(current), current, build_word(processed, 0, current));
				}
				
				// ripristino condizione precedente
				remaining.add(c);
				current--;
			}
		}
	}
	
	public void start(String phrase) {
		remaining = build_multiset(phrase);
		soluzioni = new LinkedList<>();
		backtrack(new char[remaining.size()], 0, 0, 
				remaining.size(), "");
	}
	
	public static void main(String args[]) {
		Anagrammi anagrammi = new Anagrammi();
		anagrammi.start("mangio cibo");
		int i = 0;
		for(String sol : anagrammi.soluzioni) {
			System.out.printf("%d: %s\n", ++i, sol);
		}
	}
}
