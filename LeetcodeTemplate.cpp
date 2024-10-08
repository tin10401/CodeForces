//████████╗██╗███╗░░██╗  ██╗░░░░░███████╗
//╚══██╔══╝██║████╗░██║  ██║░░░░░██╔════╝
//░░░██║░░░██║██╔██╗██║  ██║░░░░░█████╗░░
//░░░██║░░░██║██║╚████║  ██║░░░░░██╔══╝░░
//░░░██║░░░██║██║░╚███║  ███████╗███████╗
//░░░╚═╝░░░╚═╝╚═╝░░╚══╝  ╚══════╝╚══════╝
//   __________________
//  | ________________ |
//  ||          ____  ||
//  ||   /\    |      ||
//  ||  /__\   |      ||
//  || /    \  |____  ||
//  ||________________||
//  |__________________|
//  \###################\
//   \###################\
//    \        ____       \
//     \_______\___\_______\
// An AC a day keeps the doctor away.

#pragma GCC optimize("Ofast")
#pragma GCC optimize ("unroll-loops")
#pragma GCC target("popcnt")
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
template<class T> using ordered_set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
#define vt vector
#define all(x) begin(x), end(x)
#define allr(x) rbegin(x), rend(x)
#define ub upper_bound
#define lb lower_bound
#define db double
#define ld long db
#define ll long long
#define pll pair<ll, ll>    
#define vll vt<ll>  
#define vpll vt<pll>
#define vi vector<int>
#define pii pair<int, int>
#define vpii vector<pair<int, int>>
#define vs vector<string>
#define vb vector<bool>
#define vvpii vector<vpii>
#define vvi vector<vi>
#define vd vector<db>
#define ar(x) array<int, x>
#define var(x) vector<ar(x)>
#define pq priority_queue
#define mset(m, v) memset(m, v, sizeof(m))
#define pb push_back
#define ff first
#define ss second
#define sv string_view
#define MP make_pair
#define MT make_tuple
#define rsz resize
#define sum(x) accumulate(all(x), 0LL)
#define srt(x) sort(all(x))
#define srtR(x) sort(allr(x))
#define srtU(x) sort(all(x)), (x).erase(unique(all(x)), (x).end())
#define rev(x) reverse(all(x))
#define gcd(a, b) __gcd(a, b)
#define lcm(a, b) (a * b) / gcd(a, b)
#define MAX(a) *max_element(all(a)) 
#define MIN(a) *min_element(all(a))

//SGT DEFINE
#define lc i * 2 + 1
#define rc i * 2 + 2
#define lp lc, left, middle
#define rp rc, middle + 1, right
#define entireTree 0, 0, n - 1
#define midPoint left + (right - left) / 2
#define pushDown push(i, left, right)
#define iterator int i, int left, int right

#define IOS ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0)
    
//FW TREE   
#define goUp id += (id & -id)   
#define goDown id -= (id & -id)

struct custom {
    static const uint64_t C = 0x9e3779b97f4a7c15; const uint32_t RANDOM = std::chrono::steady_clock::now().time_since_epoch().count();
    size_t operator()(uint64_t x) const { return __builtin_bswap64((x ^ RANDOM) * C); }
    size_t operator()(const std::string& s) const { size_t hash = std::hash<std::string>{}(s); return hash ^ RANDOM; } };
template <class K, class V> using umap = std::unordered_map<K, V, custom>; template <class K> using uset = std::unordered_set<K, custom>;
    
 
template<typename T> vt<T> uniqued(vt<T> arr) {  srtU(arr); return arr; }

#ifdef LOCAL
#define debug(x...) debug_out(#x, x)
void debug_out(const char* names) { std::cerr << std::endl; }
template <typename T, typename... Args>
void debug_out(const char* names, T value, Args... args) {
    const char* comma = strchr(names, ',');
    std::cerr << "[" << (comma ? std::string(names, comma) : names) << " = " << value << "]";
    if (sizeof...(args)) { std::cerr << ", "; debug_out(comma + 1, args...); }   
    else { std::cerr << std::endl; }
}
#define startClock clock_t tStart = clock();
#define endClock std::cout << std::fixed << std::setprecision(10) << "\nTime Taken: " << (double)(clock() - tStart) / CLOCKS_PER_SEC << " seconds" << std::endl;
#else
#define debug(...)
#define startClock
#define endClock

#endif
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

#define eps 1e-9
#define M_PI 3.14159265358979323846
const static ll INF = 1LL << 60;
const static int MK = 20;
const static int MX = 2e6 + 5;
const static int MOD = 1e9 + 7;
int pct(ll x) { return __builtin_popcountll(x); }
const vvi dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}}; // UP, DOWN, LEFT, RIGHT
const vpii dirs_3_3 = { 
        {0,1}, {0,3},
        {1,0}, {1,2}, {1,4},
        {2,1}, {2,5},
        {3,0}, {3,4}, {3,6},
        {4,1}, {4,3}, {4,5}, {4,7},
        {5,2}, {5,4}, {5,8},
        {6,3}, {6,7},
        {7,4}, {7,6}, {7,8},
        {8,5}, {8,7}
};

int modExpo(ll base, ll exp, ll mod) { ll res = 1; base %= mod; while(exp) { if(exp & 1) res = (res * base) % mod; base = (base * base) % mod; exp >>= 1; } return res; }
void multiply(int f[2][2], int m[2][2]) {   
    int res[2][2] = {}; 
    for(int i = 0; i < 2; i++)  {   for(int j = 0; j < 2; j++)  {   for(int k = 0; k < 2; k++)  {   res[i][j] = (res[i][j] + f[i][k] * m[k][j]) % MOD; }   }   }   
    for(int i = 0; i < 2; i++)  {   for(int j = 0; j < 2; j++) f[i][j] = res[i][j]; }   }
int fib(int n)  {       if(n == 0) return 0;        if(n == 1) return 1;    
    int f[2][2] = {{1, 1}, {1, 0}}; int res[2][2] = {{1, 0}, {0, 1}};       
    while(n)    {   if(n & 1) multiply(res, f); multiply(f, f); n >>= 1;    }   return res[0][1] % MOD; }   
vi primes;  
bitset<MX> primeBits;
void generatePrime() {  primeBits.set(2);   
    for(int i = 3; i < MX; i += 2) primeBits.set(i);
    for(int i = 3; i * i < MX; i += 2) {    if(primeBits[i]) {  for(int j = i; j * i < MX; j += 2) {    primeBits.reset(i * j); } } }
    for(int i = 0; i < MX; i++ ) {  if(primeBits[i]) {  primes.pb(i); } }   
}

vi countBit(ll n) { 
    vi cnt(62);
    while(n) {  
        int msb = log2(n);  
        ll c = 1LL << msb; 
        cnt[msb] += n - c + 1;
        for(int i = 0; i < msb; i++) {  
            cnt[i] += c / 2;
        }
        n -= c;
    }
    return cnt;
}
    
template<typename T>
class Treap {
private:
    struct TreapNode {
        int pri, size;
        T data;
        TreapNode* left;
        TreapNode* right;
        
        TreapNode(T data) : data(data), pri(rand()), size(1), left(nullptr), right(nullptr) {}
    };

    TreapNode* root;

    int size(TreapNode* treap) {
        if (!treap) return 0;
        return treap->size;
    }

    void split(TreapNode* treap, TreapNode*& left, TreapNode*& right, int k) {
        if (!treap) {
            left = right = nullptr;
            return;
        }
        if (size(treap->left) >= k) {
            split(treap->left, left, treap->left, k);
            right = treap;
        } else {
            split(treap->right, treap->right, right, k - size(treap->left) - 1);
            left = treap;
        }
        treap->size = size(treap->left) + size(treap->right) + 1;
    }

    void merge(TreapNode*& treap, TreapNode* left, TreapNode* right) {
        if (!left || !right) {
            treap = left ? left : right;
            return;
        }
        if (left->pri < right->pri) {
            merge(left->right, left->right, right);
            treap = left;
        } else {
            merge(right->left, left, right->left);
            treap = right;
        }
        treap->size = size(treap->left) + size(treap->right) + 1;
    }

public:
    Treap() : root(nullptr) {}

    void insert(T ch) { 
        merge(root, root, new TreapNode(ch));
    }
    
    void del(int left, int right) { 
        TreapNode* A, *B, *C;   
        split(root, A, B, left - 1); 
        split(B, B, C, right - left + 1);   
        merge(root, A, C);
    }
    
    bool get(TreapNode* treap, int k, T& ans) {
        if(!treap) return true;
        int leftSize = size(treap->left);
        if (k <= leftSize) {
            if(get(treap->left, k, ans)) {  
                ans = treap->data;
            }
            return false;
        }
        else if (k == leftSize + 1) {
            ans = treap->data;  
            return false;
        }
        else {
            if(get(treap->right, k - leftSize - 1, ans)) {  
                ans = treap->data;
            }
            return false;
        }
    }
};
    
class Binary_Trie { 
    public:
    int T[MX][2];   
    int ptr;    
    Binary_Trie() {    
        ptr = 0;    
        mset(T, 0);
    }
    
    void insert(int num) {  
        int curr = 0;   
        for(int i = 31; i >= 0; i--) {  
            int bits = (num >> i) & 1;  
            if(!T[curr][bits]) T[curr][bits] = ++ptr;   
            curr = T[curr][bits];
        }
    }
        
    int max_xor(int num) {  
        int res = 0, curr = 0;
        for(int i = 31; i >= 0; i--) {  
            int bits = (num >> i) & 1;  
            if(T[curr][!bits]) {    
                curr = T[curr][!bits];
                res |= (1LL << i);
            }
            else {  
                curr = T[curr][bits];
            }
        }
        return res;
    }
};

class Trie {
private:
    int root;
    int count = 0, n;
    vector<int> dp;
    string s;
    int T[MX][26];       
    int sfx[MX];         
    int dict[MX];        
    int id[MX];          
    bool isEnd[MX];


    int newNode() {
        fill(T[count], T[count] + 26, -1); 
        sfx[count] = dict[count] = 0;      
        id[count] = -1; 
        isEnd[count] = false;                    
        return count++;                     
    }

public:
    Trie(int n) {
        this->n = n;
        dp.rsz(n + 1, 1e9);
        root = newNode(); 
        sfx[root] = dict[root] = root; 
    }

    void insert(const string& word) {
        int curr = root;
        int l = 0;
        for (char ch : word) {
            int idx = ch - 'a';
            if (T[curr][idx] == -1) {
                T[curr][idx] = newNode();
            }
            curr = T[curr][idx];
            l++;
            if (id[curr] == -1) {
                id[curr] = l; 
            }
        }
        isEnd[curr] = true;
    }

    void aho_corasick() {
        queue<int> q;
        q.push(root);

        while (!q.empty()) {
            int par = q.front();
            q.pop();

            for (int i = 0; i < 26; i++) {
                int child = T[par][i];
                if (child == -1) continue;

                int suff = sfx[par];
                while (suff != root && T[suff][i] == -1) {
                    suff = sfx[suff];
                }

                if (par != root && T[suff][i] != -1) {
                    sfx[child] = T[suff][i];
                } else {
                    sfx[child] = root;
                }

                dict[child] = (id[sfx[child]] == -1) ? dict[sfx[child]] : sfx[child];
                q.push(child);
            }
        }
    }

    void queries(int& prev, int i, char ch) {
        int idx = ch - 'a';
        while (prev != root && T[prev][idx] == -1) {
            prev = sfx[prev];
        }

        if (T[prev][idx] != -1) {
            prev = T[prev][idx];
            int curr = (id[prev] == -1) ? dict[prev] : prev;

            while (id[curr] != -1) {
                int j = id[curr];
                dp[i] = min(dp[i], dp[i - j] + 1);
                curr = dict[curr];
            }
        }
    }

    int get() {
        dp[0] = 0;
        int prev = root;

        for (int i = 1; i <= n; i++) {
            queries(prev, i, s[i - 1]);
        }

        return dp[n] == 1e9 ? -1 : dp[n];
    }
};

class DSU { 
    public: 
    int n;  
    vi root, rank;  
    DSU(int n) {    
        this->n = n;    
        root.rsz(n, -1), rank.rsz(n, 1);
    }
    
    int find(int x) {   
        if(root[x] == -1) return x; 
        return root[x] = find(root[x]);
    }
    
    bool merge(int u, int v) {  
        u = find(u), v = find(v);   
        if(u != v) {    
            if(rank[v] > rank[u]) swap(u, v);   
            rank[u] += rank[v]; 
            root[v] = u;
            return true;
        }
        return false;
    }
    
    bool same(int u, int v) {    
        return find(u) == find(v);
    }
    
    int getRank(int x) {    
        return rank[find(x)];
    }
};
    
class FW {  
    public: 
    int n;  
    vi root;    
    FW(int n) { 
        this->n = n;    
        root.rsz(n + 1);
    }
    
    void update(int id, int val) {  
        while(id <= n) {    
            root[id] += val;    
            id += (id & -id);
        }
    }
    
    int get(int id) {   
        int res = 0;    
        while(id > 0) { 
            res += root[id];    
            id -= (id & -id);
        }
        return res;
    }
    
    int queries(int left, int right) {  
        return get(right) - get(left - 1);
    }
};

template<class T>   
class SGT { 
    public: 
    int n;  
    vt<T> root, lazy; 
    T DEFAULT;
    SGT(vi& arr) {    
        n = arr.size(); 
        DEFAULT = INF;
        root.rsz(n * 4);    
        // lazy.rsz(n * 4);
        build(entireTree, arr);
    }
    
    void build(iterator, vi& arr) { 
        if(left == right) { 
            root[i] = arr[left];    
            return;
        }
        int middle = midPoint;  
        build(lp, arr), build(rp, arr); 
        root[i] = merge(root[lc], root[rc]);
    }
    
    void update(int id, int val) {  
        update(entireTree, id, val);
    }
    
    void update(iterator, int id, int val) {    
        if(left == right) { 
            root[i] = val;  
            return;
        }
        int middle = midPoint;  
        if(id <= middle) update(lp, id, val);   
        else update(rp, id, val);   
        root[i] = merge(root[lc], root[rc]);
    }

    void update(int start, int end, int val) { 
        update(entireTree, start, end, val);
    }
    
    void update(iterator, int start, int end, int val) {    
        pushDown;   
        if(left > end || start > right) return; 
        if(left >= start && right <= end) { 
            lazy[i] = val;  
            pushDown;   
            return;
        }
        int middle = midPoint;  
        update(lp, start, end, val);    
        update(rp, start, end, val);    
        root[i] = merge(root[lc], root[rc]);
    }
    
    T merge(T left, T right) {  
        T res;  
        return res;
    }
    
    void push(iterator) {   
        if(lazy[i] == 0) return;    
        root[i] += (right - left + 1) * lazy[i];
        if(left != right) { 
            lazy[lc] += lazy[i]; 
            lazy[rc] += lazy[i];
        }
        lazy[i] = 0;
    }

    T queries(int start, int end) { 
        return queries(entireTree, start, end);
    }
    
    T queries(iterator, int start, int end) {   
        pushDown;
        if(left > end || start > right) return DEFAULT;
        if(left >= start && right <= end) return root[i];   
        int middle = midPoint;  
        return merge(queries(lp, start, end), queries(rp, start, end));
    }

};
    
vi KMP(const string& s) {   
    int n = s.size();
    vi prefix(n);
    for(int i = 1, j = 0; i < n; i++) { 
        while(j && s[i] != s[j]) j = prefix[j - 1]; 
        if(s[i] == s[j]) prefix[i] = ++j;
    }
    return prefix;
}

vi Z_Function(const string& s) {    
    int n = s.size();   
    vi prefix(n);   
    for(int i = 1, left = 0, right = 0; i < n; i++) {   
        if(i > right) { 
            left = right = i;   
            while(right < n && s[right] == s[right - left]) right++;    
            prefix[i] = right-- - left;
        }
        else {  
            if(prefix[i - left] + i < right + 1) {  
                prefix[i] = prefix[i - left];
            }
            else {  
                left = i;   
                while(right < n && s[right] == s[right - left]) right++;    
                prefix[i] = right-- - left;
            }
        }
    }
    return prefix;
}
    
vi manacher(string s, int start) {
    string tmp;
    for (auto& it : s) {
        tmp += "#";
        tmp += it;
    }
    tmp += "#";  
    swap(s, tmp);
    int n = s.size();
    vector<int> p(n); 
    int l = 0, r = 0;  
    for (int i = 0; i < n; i++) {
        if (i < r) {
            p[i] = min(r - i, p[l + r - i]);
        } else {
            p[i] = 0;
        }
        while (i - p[i] >= 0 && i + p[i] < n && s[i - p[i]] == s[i + p[i]]) {
            p[i]++;
        }
        if (i + p[i] > r) {
            l = i - p[i] + 1;
            r = i + p[i] - 1;
        }
    }
    vi result;
    for (int i = start; i < n; i += 2) {
        result.push_back(p[i] / 2);
    }
    return result;
}

class RabinKarp {   
    public: 
    vpii prefix;    
    vi pow;
    int mod1, n, mod2, base1, base2;
    RabinKarp(const string& s) {  
        mod1 = 1e9 + 7, mod2 = 1e9 + 33, base1 = 26, base2 = 27;
        n = s.size(); 
        prefix.rsz(n + 1);
        pow.rsz(n + 1);
        pow[0] = 1;
        buildHash(s);
    }
    
    void buildHash(const string& s) {   
        int hash1 = 0, hash2 = 0;
        for(int i = 1; i <= n; i++) {   
            hash1 = (hash1 * base1 + s[i - 1] - 'a') % mod1;    
            hash2 = (hash2 * base2 + s[i - 1] - 'a') % mod2;
            prefix[i].ff = hash1;    
            prefix[i].ss = hash2;
            pow[i] = (pow[i - 1] * 26) % mod1;
        }
    }
    
    pii getHash(int l, int r) { 
        int hash1 = prefix[r].ff - (prefix[l].ff * pow[r - l] % mod1) % mod1;
        hash1 = (hash1 + mod1) % mod1;  
        int hash2 = prefix[r].ss - (prefix[l].ss * pow[r - l] % mod2) % mod2;   
        hash2 = (hash2 * mod2) % mod2;  
        return MP(hash1, hash2);
    };
};

class LCA { 
    public: 
    int n;  
    vvi dp; 
    vi depth;
    LCA(vvi& dp, vi& depth) {   
        this->dp = dp;  
        this->depth = depth;
        n = depth.size();
        init();
    }
    
    void init() {  
        for(int j = 1; j < MK; j++) {   
            for(int i = 0; i < n; i++) {    
                dp[i][j] = dp[dp[i][j - 1]][j - 1];
            }
        }
    }
    
    int lca(int a, int b) { 
        if(depth[a] > depth[b]) {   
            swap(a, b);
        }
        int d = depth[b] - depth[a];    
        for(int i = MK - 1; i >= 0; i--) {  
            if((d >> i) & 1) {  
                b = dp[b][i];
            }
        }
        if(a == b) return a;    
        for(int i = MK - 1; i >= 0; i--) {  
            if(dp[a][i] != dp[b][i]) {  
                a = dp[a][i];   
                b = dp[b][i];
            }
        }
        return dp[a][0];
    }
};

class Combinatoric {    
    public: 
    int n;  
    vll fact, inv;   
    Combinatoric(int n) {   
        this->n = n;    
        fact.rsz(n + 1), inv.rsz(n + 1);
        init();
    }
        
    void init() {   
        fact[0] = 1;
        for(int i = 1; i <= n; i++) {   
            fact[i] = (fact[i - 1] * i) % MOD;
        }
        inv[n] = modExpo(fact[n], MOD - 2, MOD);
        for(int i = n - 1; i >= 0; i--) {   
            inv[i] = (inv[i + 1] * (i + 1)) % MOD;
        }
    }
    
    int choose(int a, int b) {  
        return fact[a] * inv[b] % MOD * inv[a - b] % MOD;
    }
};
