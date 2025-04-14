#include <bits/stdc++.h>
using namespace std;
 
// We'll use Mersenne Twister for randomness.
mt19937 rng((unsigned)chrono::steady_clock::now().time_since_epoch().count());
 
// Generate a random integer in the inclusive range [l, r].
int randInt(int l, int r) {
    uniform_int_distribution<int> uid(l, r);
    return uid(rng);
}
 
// Generates a random tree on n vertices (1-indexed)
// using the "random parent" method: for i=2..n, choose a parent uniformly from [1,i-1].
vector<pair<int,int>> generateTree(int n) {
    vector<pair<int,int>> edges;
    for (int i = 2; i <= n; i++) {
        int parent = randInt(1, i - 1);
        edges.push_back({parent, i});
    }
    // Optionally, shuffle the edges
    shuffle(edges.begin(), edges.end(), rng);
    return edges;
}
 
// Generates a random set of s distinct shops (1-indexed) from 1..n.
vector<int> generateShops(int n, int s) {
    vector<int> nodes(n);
    iota(nodes.begin(), nodes.end(), 1);
    shuffle(nodes.begin(), nodes.end(), rng);
    vector<int> shops(nodes.begin(), nodes.begin() + s);
    sort(shops.begin(), shops.end());
    return shops;
}
 
// Main test case generator.
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    // You can change these default values or read from command-line arguments.
    int n = 10; // number of intersections
    int q = 10;  // number of customers
    // For shops, choose a random number between 1 and n:
    int s = randInt(1, n);
    
    // Print first line: n and q
    cout << n << " " << q << "\n";
    
    // Generate and output a random tree.
    vector<pair<int,int>> treeEdges = generateTree(n);
    for (auto &e : treeEdges) {
        // Output each edge (u v), 1-indexed.
        cout << e.first << " " << e.second << "\n";
    }
    
    // Generate shop list.
    vector<int> shops = generateShops(n, s);
    // Output the shop list in one line.
    for (int i = 0; i < s; i++) {
        cout << shops[i] << (i + 1 < s ? " " : "\n");
    }
    
    // For each customer, generate a favorite shop (chosen from shops) and a patience value.
    // For the patience value, we choose a random value between 0 and, say, n.
    for (int i = 0; i < q; i++) {
        int fav = shops[randInt(0, s - 1)];
        int patience = randInt(0, n);
        cout << fav << " " << patience << "\n";
    }
    
    return 0;
}

