---
layout: post
title:  "Solution of 'The Algorithm Design Manual' 2nd edition"
date:   2017-04-04 14:16:00 +0200
categories: notes
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

[The Algorithm Design Manual][buy-book] is the book I use in my algorithm course. I prefer this one to the [other, universally used book about algoritms][shit]. I'm posting here some of the solutions of the exercises I've done. All exercises are written in C or Java. I didn't copy a single line of code from the book or anywhere else, with the exception of the structure of `Backtrackable` class. All the images are made in Latex (the trees are made with `forest` package). All the codes (should) at least compile, most of them should work. Some of the explanations are in Italian (I will translate them).

* TOC
{:toc}

Chapter 7: Combinatorial Search and Heuristic Methods
-----------------------------------------------------

This is the base class that any execise must inherit:
{% highlight Java %}
{% include algorithms/Backtrackable.java %}
{% endhighlight %}

Exercise 7.1: write all dearrangements
======================================

A dearrangement is a permutation where i-th number does not occur in i-th position. The sequence $${3, 1, 2}$$ is a dearrangement of $${1, 2, 3}$$, while $${3, 2, 1}$$ is not. Here's the recursion tree for n=3:

![Exercise 7.1 Recursion Tree][ex-71-image]{: .center-image }

{% highlight Java %}
{% include algorithms/ConstructAllDearrangement.java %}
{% endhighlight %}

Exercise 7.2: write all multiset permutations
=============================================

Multisets may have repeated elements. For this reason, it may have less than $n!$ permutation. The set $${1, 1, 1, 1, 2, 2, 2}$$ has the 1 repeated four times, and the 2 repeated three times; the number of permutation of this multiset are $$\frac{n!}{r_1! r_2!} = \frac{7!}{4! 3!} = \frac{5040}{24 \cdot 6} = 35$$.

![Exercise 7.2 Recursion Tree][ex-72-image]{: .center-image }

{% highlight Java %}
{% include algorithms/ConstructAllMultiset.java %}
{% endhighlight %}

This second version uses `Multiset` class from Google's `Guava`

{% highlight Java %}
{% include algorithms/ConstructAllMultiset2.java %}
{% endhighlight %}

Exercise 7.4: anagrams
======================

Given a dictionary, write all anagrams of an input string.

Naive strategy: write all permutations of the string, split it in any possible way, then check all the word are in the dictionary.

We can improve efficiency using a data structure called [Trie][trie]. It is a dictionary of string that has the operation `startWith(String prefix)` that return `true` iff exists a word in the dictionary starting with the given prefix.

![Trie][trie-img]{: .center-image }

**Explanation (in Italian)**

![Exercise 7.4 Recursion Tree][ex-74-image]{: .center-image }

**Explanation (in English)**

Todo

{% highlight Java %}
public class Trie {

    public Trie() { ... }

    // Inserts a word into the trie.
    public void insert(String word) { ... }

    // Returns if the word is in the trie.
    public boolean contains(String word) { ... }

    // Returns if there is any word in the trie
    // that starts with the given prefix.
    public boolean startsWith(String prefix) {...}
}

{% include algorithms/Anagrammi.java %}
{% endhighlight %}


[buy-book]: https://www.amazon.com/Algorithm-Design-Manual-Steven-Skiena/dp/1848000693
[shit]: https://www.amazon.com/Introduction-Algorithms-3rd-MIT-Press/dp/0262033844/
[ex-71-image]: {{ site.url }}/assets/algorithms/dearrangement.png
[ex-72-image]: {{ site.url }}/assets/algorithms/multiset.png
[ex-74-image]: {{ site.url }}/assets/algorithms/anagrammi.png
[trie]: https://en.wikipedia.org/wiki/Trie
[trie-img]: https://upload.wikimedia.org/wikipedia/commons/b/be/Trie_example.svg

