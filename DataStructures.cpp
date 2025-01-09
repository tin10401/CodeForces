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
#define ll int64_t
#define vll vt<ll>  
#define vvll vt<vll>
#define pll pair<ll, ll>    
#define vpll vt<pll>
#define vc vt<char> 
#define vvc vt<vc>
#define vi vt<int>
#define vvi vt<vi>
#define vvvi vt<vvi>
#define pii pair<int, int>
#define vpii vt<pii>
#define vs vt<string>
#define vvs vt<vs>
#define vb vt<bool>
#define vvb vt<vb>
#define vvpii vt<vpii>
#define vd vt<db>
#define ar(x) array<int, x>
#define var(x) vt<ar(x)>
#define vvar(x) vt<var(x)>
#define pq priority_queue
#define mset(m, v) memset(m, v, sizeof(m))
#define pb push_back
#define ff first
#define ss second
#define sv string_view
#define MP make_pair
#define MT make_tuple
#define rsz resize
#define sum(x) (ll)accumulate(all(x), 0LL)
#define srt(x) sort(all(x))
#define srtR(x) sort(allr(x))
#define srtU(x) sort(all(x)), (x).erase(unique(all(x)), (x).end())
#define SORTED(x) is_sorted(all(x))
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
#define iter int i, int left, int right

#define IOS ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0)


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
const int inf = 1e9 + 33;
const static int MK = 20;
const static int MX = 2e6 + 5;
const static int MOD = 1e9 + 7;
int pct(ll x) { return __builtin_popcountll(x); }
const vvi dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}}; // UP, DOWN, LEFT, RIGHT

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
    
template<typename T>
class Treap {
private:
    struct TreapNode {
        int pri, size, f;
        T key;
        TreapNode* left;
        TreapNode* right;
        
        TreapNode(T key) : f(0), key(key), pri(rand()), size(1), left(nullptr), right(nullptr) {}
    };

    TreapNode* root;

    int size(TreapNode* treap) {
        if (!treap) return 0;
        return treap->size;
    }
    
    void unite(TreapNode* treap) {  
        treap->size = size(treap->left) + size(treap->right) + 1;
    }

    T flip(T bit) { 
        return !bit;
    }

    void apply(TreapNode* treap) {  
        if(!treap) return;
        if(treap->f) {  
            swap(treap->left, treap->right);
            if(treap->left) treap->left->f ^= 1;    
            if(treap->right) treap->right->f ^= 1;
            treap->f = 0;
        }
    }

    void split(TreapNode* treap, TreapNode*& left, TreapNode*& right, int k) {
        if (!treap) {
            left = right = nullptr;
            return;
        }
        apply(treap);
        if (size(treap->left) >= k) {
            split(treap->left, left, treap->left, k);
            right = treap;
        } else {
            split(treap->right, treap->right, right, k - size(treap->left) - 1);
            left = treap;
        }
        unite(treap);
    }

    void merge(TreapNode*& treap, TreapNode* left, TreapNode* right) {
        if (!left || !right) {
            treap = left ? left : right;
            return;
        }
        apply(treap);
        if (left->pri < right->pri) {
            merge(left->right, left->right, right);
            treap = left;
        } else {
            merge(right->left, left, right->left);
            treap = right;
        }
        unite(treap);
    }

    void destroy(TreapNode* treap) {
        if (!treap) return;
        destroy(treap->left);
        destroy(treap->right);
        delete treap;
    }

public:
    Treap() : root(nullptr) {}
    
    ~Treap() {
        destroy(root);
    }

    void insert(T key) { 
        merge(root, root, new TreapNode(key));
    }
    
	void split_and_reverse(int l, int r) { // off_set by 1
        TreapNode* A, *B;
        split(root, root, A, l - 1); 
        split(A, A, B, r - l + 1);
        if(A) { 
            A->f ^= 1;  
            apply(A);
        }
        merge(root, root, A);   
        merge(root, root, B);
    }
	
	void split_and_swap(int k) { // off_set by 1
        if(k == 0 || k == size(root)) return; 
        TreapNode* A, *B, *C;
        split(root, root, A, k);
        if(!A) return;
        merge(root, A, root);
    }

    void shift_right(int l, int r) { // [1 2 3 4] -> [4 1 2 3] and off_set by 1
        r = r - l + 1;
        TreapNode* A, *B, *C;
        split(root, root, A, l - 1);
        split(A, A, B, r);
        split(A, A, C, r - 1);
        merge(root, root, C);
        merge(root, root, A);
        merge(root, root, B);
    }

    T get(int k) {  
        TreapNode* A, *B, *C;   
        split(root, A, B, k - 1);
        split(B, B, C, 1);  
        T ans = B->key;
        merge(root, A, B);  
        merge(root, root, C);
        return ans;
    }
	
	TreapNode* erase(int l, int r) {
        TreapNode* A, *B;
        split(root, root, A, l - 1);
        split(A, A, B, r);
        merge(root, root, B);
        return A;
    }

    TreapNode* merge_treap(TreapNode* A, TreapNode* B) {
        if(!A) return B;
        if(!B) return A;
        if(A->pri < B->pri) swap(A, B);
        TreapNode*L, *R;
        split(B, L, R, A->key);
        A->left = merge_treap(L, A->left);
        A->right = merge_treap(A->right, R);
        return A;
    }

    void merge_treap(TreapNode* other) {
        root = merge_treap(root, other);
    }

    
    void print() {  
        print(root);
        cout << endl;
    }
	
    void print(TreapNode* treap) {  
        apply(treap);
        if(!treap) return;
        print(treap->left); 
        cout << treap->val;
        print(treap->right);
    }
};
    
template<class T>
class FW {  
    public: 
    int n, N;
    vt<T> root;    
    FW(int n) { 
        this->n = n;    
        N = log2(n);
        root.rsz(n);
    }
    
    void update(int id, T val) {  
        while(id < n) {    
            root[id] += val;    
            id |= (id + 1);
        }
    }
    
    T get(int id) {   
        T res = 0;    
        while(id >= 0) { 
            res += root[id];    
            id = (id & (id + 1)) - 1;
        }
        return res;
    }
    
    T queries(int left, int right) {  
        return get(right) - get(left - 1);
    }
	
	void reset() {
		root.assign(n, 0);
	}

    int search(int x) { // get pos where sum >= x
        int global = get(n), curr = 0;
        for(int i = N; i >= 0; i--) {
            int t = curr ^ (1LL << i);
            if(t < n && global - root[t] >= x) {
                swap(curr, t);
                global -= root[curr];
            }
        }
        return curr + 1;
    }
};


template<class T>   
class SGT { 
    public: 
    int n;  
    vt<T> root;
	vll lazy;
    T DEFAULT;
	SGT(int n, T DEFAULT) {    
        this->n = n;
        this->DEFAULT = DEFAULT;
        root.rsz(n * 4);    
        lazy.rsz(n * 4);
//        build(entireTree, arr);
    }
    
//    void build(iter, vi& arr) { 
//        if(left == right) { 	
//            root[i] = arr[left];    
//            return;
//        }
//        int middle = midPoint;  
//        build(lp, arr), build(rp, arr); 
//        root[i] = merge(root[lc], root[rc]);
//    }

    
    void update(int id, T val) {  
        update(entireTree, id, val);
    }
    
    void update(iter, int id, T val) {  
		pushDown;
        if(left == right) { 
            root[i] = val;  
            return;
        }
        int middle = midPoint;  
        if(id <= middle) update(lp, id, val);   
        else update(rp, id, val);   
        root[i] = merge(root[lc], root[rc]);
    }

    void update(int start, int end, T val) { 
        update(entireTree, start, end, val);
    }
    
    void update(iter, int start, int end, T val) {    
        pushDown;   
        if(left > end || start > right) return; 
        if(left >= start && right <= end) { 
			apply(i, left, right, val);
            pushDown;   
            return;
        }
        int middle = midPoint;  
        update(lp, start, end, val);    
        update(rp, start, end, val);    
        root[i] = merge(root[lc], root[rc]);
    }
    
	void apply(iter, T val) {
        root[i] += val;
        lazy[i] += val;
    }

    void push(iter) {   
        if(lazy[i] && left != right) {
			int middle = midPoint;
            apply(lp, lazy[i]), apply(rp, lazy[i]);
            lazy[i] = 0;
        }
    }

	T queries(int id) {
		return queries(entireTree, id);
	}
	
	T queries(iter, int id) {
		pushDown;
		if(left == right) {
			return root[i];
		}
		int middle = midPoint;
		if(id <= middle) return queries(lp, id);
		return queries(rp, id);
	}

    T queries(int start, int end) { 
        return queries(entireTree, start, end);
    }
    
    T queries(iter, int start, int end) {   
        pushDown;
        if(left > end || start > right) return DEFAULT;
        if(left >= start && right <= end) return root[i];   
        int middle = midPoint;  
        return merge(queries(lp, start, end), queries(rp, start, end));
    }
	
	T get() {
		return root[0];
	}
	
	void print() {  
        print(entireTree);
        cout << endl;
    }
    
    void print(iter) {  
        pushDown;
        if(left == right) { 
            cout << root[i] << ' ';
            return;
        }
        int middle = midPoint;  
        print(lp);  print(rp);
    }

    T merge(T left, T right) {  
        T res;  
        res = max(left, right);
        return res;
    }
};



// PERSISTENT SEGTREE
int T[MX * MK], root[MX * MK * 4], ptr; 
pii child[MX * MK * 4];
void update(int curr, int prev, int id, int delta, int left, int right) {  
    root[curr] = root[prev];    
    child[curr] = child[prev];
    if(left == right) { 
        root[curr] += delta;
        return;
    }
    int middle = midPoint;
    if(id <= middle) {  
        child[curr].ff = ++ptr; 
        update(child[curr].ff, child[prev].ff, id, delta, left, middle);
    }
    else {  
        child[curr].ss = ++ptr; 
        update(child[curr].ss, child[prev].ss, id, delta, middle + 1, right);
    }
    root[curr] = root[child[curr].ff] + root[child[curr].ss];
}

ll queries(int curr, int prev, int start, int end, int left, int right) { 
    if(left >= start && right <= end) return root[curr] - root[prev];
    if(left > end || start > right) return 0;
    int middle = midPoint;  
    return queries(child[curr].ff, child[prev].ff, start, end, left, middle) + queries(child[curr].ss, child[prev].ss, start, end, middle + 1, right);
};
    
int get(int curr, int prev, int k, int left, int right) {    
    if(root[curr] - root[prev] < k) return inf;
    if(left == right) return left;
    int leftCount = root[child[curr].ff] - root[child[prev].ff];
    int middle = midPoint;
    if(leftCount >= k) return get(child[curr].ff, child[prev].ff, k, left, middle);
    return get(child[curr].ss, child[prev].ss, k - leftCount, middle + 1, right);
}

void reset() {  
    for(int i = 0; i <= ptr; i++) { 
        root[i] = 0, T[i] = 0;  
        child[i] = MP(0, 0);
    }
	ptr = 0;
}

class MO {  
    public: 
    int n, q;  
    int block;
    vi a;   
    var(3) Q;
    MO(vi& a, var(3)& Q) {  // 1 base index array
        n = a.size();
        q = Q.size();
        this->a = a;    
        this->Q = Q;
        block = sqrt(n);
    }

    vll queries() {    
        auto cmp = [&](const ar(3)& a, const ar(3)& b) -> bool {    
            if(a[0] / block != b[0] / block) return a[0] / block < b[0] / block;
            int d = a[0] / block;   
            if(d & 1) return a[1] > b[1];
            return a[1] < b[1];
        };
        sort(all(Q), cmp);
        vi pos(a);  
        srtU(pos); 
        umap<int, int> mp;  
        int N = pos.size();
        for(int i = 0; i < N; i++) mp[pos[i]] = i;
        for(auto& it : a) it = mp[it];

        vll dp(N);
        ll ans = 0;
        auto modify = [&](int x, int v) -> void {    
            if(pos[x] == 0) return;
            if(dp[x] == pos[x]) ans--;  
            dp[x] += v; 
            if(dp[x] == pos[x]) ans++;
        };

        vll res(q);
        int l = 0, r = -1;    // modify to 0 as needed "left = 0"
        for(auto& [l, r, id] : Q) { 
			while (r < qr) modify(a[++r], 1);
			while (l > ql) modify(a[--l], 1);
			while (r > qr) modify(a[r--], -1);
			while (l < ql) modify(a[l++], -1);
            res[id] = ans;
        }
        return res;
    }
};

template<class T>
class SparseTable { 
    public: 
    int n;  
    vi a, log_table;
    vt<vt<T>> dp_max, dp_min, dp_gcd, dp_or;
	SparseTable(vi& a) {  
		n = a.size();
        this->a = a;
        log_table.rsz(n + 1);
        dp_max.rsz(n, vt<T>(MK));
        dp_min.rsz(n, vt<T>(MK));
        dp_gcd.rsz(n, vt<T>(MK));
        dp_or.rsz(n, vt<T>(MK));
        init();
    }
    
    void init() {   
		for(int i = 2; i <= n; i++) log_table[i] = log_table[i / 2] + 1;
        for(int i = 0; i < n; i++) dp_max[i][0] = dp_min[i][0] = dp_gcd[i][0] = dp_or[i][0] = a[i];
        for(int j = 1; j < MK; j++) {    
            for(int i = 0; i + (1LL << j) <= n; i++) {    
                int p = i + (1LL << (j - 1));
                dp_max[i][j] = max(dp_max[i][j - 1], dp_max[p][j - 1]);
                dp_min[i][j] = min(dp_min[i][j - 1], dp_min[p][j - 1]);
                dp_gcd[i][j] = gcd(dp_gcd[i][j - 1], dp_gcd[p][j - 1]);
                dp_or[i][j] = dp_or[i][j - 1] | dp_or[p][j - 1];
            }
        }
    }
    
    int queries(int left, int right) {  
		int j = log_table[right - left + 1];
        int p = right - (1LL << j) + 1;
        T mx = max(dp_max[left][j], dp_max[p][j]);
        T mn = min(dp_min[left][j], dp_min[p][j]);
        T g = gcd(dp_gcd[left][j], dp_gcd[p][j]);
        T OR = dp_or[left][j] | dp_or[p][j];
        return OR;
    }
};



class TWO_DIMENSIONAL_RANGE_QUERY {   
    public: 
    vvll prefix;
    vvi grid;
    int n, m;
    TWO_DIMENSIONAL_RANGE_QUERY(vvi& grid) {  
        n = grid.size(), m = grid[0].size();
        this->grid = grid;
        prefix.assign(n + 1, vll(m + 1));  
        init();
    }
    
    ll get(int r1, int c1, int r2, int c2) {   
        ll bottomRight = prefix[r2][c2];   
        ll topLeft = prefix[r1 - 1][c1 - 1];
        ll topRight = prefix[r1 - 1][c2];  
        ll bottomLeft = prefix[r2][c1 - 1];
        return bottomRight - topRight - bottomLeft + topLeft;
    }

    void init() {   
         for(int i = 1; i <= n; i++) {  
             ll sm = 0;
             for(int j = 1; j <= m; j++) {  
                 sm += grid[i - 1][j - 1];
                 prefix[i][j] = sm + prefix[i - 1][j];
             }
         }
    }
};


int root[MX * 120], lazy[MX * 120], ptr; // MX should be 1e5
pii child[MX * 120];
class implit_segtree {
    public:
    int n;
    implit_segtree(int n) {
        this->n = n;
        root[0] = a.queries(0, n - 1); // initialize
        lazy[0] = -1;
    } 

    void create_node(int& node, int left, int right) {
        if(node) return;
        node = ++ptr;
        lazy[node] = -1;
        root[node] = a.queries(left, right);
    }

    void update(int start, int end, int x) {
        update(entireTree, start, end, x);
    }

    void update(iter, int start, int end, int x) {
        pushDown;
        if(left > end || start > right) return;
        if(start <= left && right <= end) {
			apply(i, left, right, x);
            pushDown;
            return;
        }
        int middle = midPoint;
        create_node(child[i].ff, left, middle);
        create_node(child[i].ss, middle + 1, right);
        update(child[i].ff, left, middle, start, end, x);
        update(child[i].ss, middle + 1, right, start, end, x);
        root[i] = merge(root[child[i].ff], root[child[i].ss]);
    }

	void apply(iter, int val) {
        root[i] = val * (right - left + 1);
        lazy[i] = val;
    }

    void push(iter) {
        if(lazy[i] != -1 && left != right) {
            int middle = midPoint;
            create_node(child[i].ff, left, middle);
            create_node(child[i].ss, middle + 1, right);
			apply(child[i].ff, left, middle, lazy[i]);
            apply(child[i].ss, middle + 1, right, lazy[i]);
            lazy[i] = -1;
        }
    }

    int merge(int left, int right) {
        return min(left, right);
    }

    int queries(int start, int end) {
        return queries(entireTree, start, end);
    }

    int queries(iter, int start, int end) {
        pushDown;
        if(start > right || left > end) return inf;
        if(start <= left && right <= end) return root[i];
        int middle = midPoint;
        create_node(child[i].ff, left, middle);
        create_node(child[i].ss, middle + 1, right);
        return merge(queries(child[i].ff, left, middle, start, end), queries(child[i].ss, middle + 1, right, start, end));
    }
	
	void update(int& i, int x) {
        update(i, 0, n - 1, x);
    }

    void update(int& i, int left, int right, int x) {
        if(!i) i = ++ptr;
        if(left == right) return;
        int middle = midPoint;
        if(x <= middle) update(child[i].ff, left, middle, x);
        else update(child[i].ss, middle + 1, right, x);
    }


    int merge_two_tree(int i, int j) {
        if(!i || !j) return i + j;
        child[i].ff = merge_two_tree(child[i].ff, child[j].ff);
        child[i].ss = merge_two_tree(child[i].ss, child[j].ss);
        return i;
    }

    void modify_two_tree(int& i, int& j, int start, int end) {
        modify(i, j, 0, n - 1, start, end);
    }

    void modify(int& i, int& j, int left, int right, int start, int end) {
        if(!i || left > end || start > right) return;
        if(!j) j = ++ptr;
        if(left >= start && right <= end) {
            j = merge_two_tree(i, j);
            i = 0;
            return;
        }
        int middle = midPoint;
        modify(child[i].ff, child[j].ff, left, middle, start, end);
        modify(child[i].ss, child[j].ss, middle + 1, right, start, end);
    }

};

template<class T>   
class SegTree_Graph { 
    public: 
    int n;  
    vvpii graph;
    vi pos;
    int off;
	SegTree_Graph(int n) {    
        this->n = n;
        off = n * 4;
        graph.rsz(n * 8 + 23);
        pos.rsz(n);
        build(entireTree);
    }
    
    void build(iter) { 
        int u = i + off; // u here means from the parent can go to the children using off_set edges
        if(left == right) {
            graph[i].pb({u, 0});
            graph[u].pb({i, 0});
            pos[left] = i;
            return;
        }
        int middle = midPoint;  
        build(lp), build(rp); 
        graph[lc].pb({i, 0});
        graph[rc].pb({i, 0});
        graph[u].pb({lc + off, 0});
        graph[u].pb({rc + off, 0});
    }

    void add_edge(int u, int v, int w) {
        u = pos[u], v = pos[v];
        graph[u].pb({v, w});
    }

    void update(int start, int end, int u, int w, int type) { 
        update(entireTree, start, end, pos[u], w, type);
    }
    
    void update(iter, int start, int end, int u, int w, int type) {    
        if(left > end || start > right) return; 
        if(left >= start && right <= end) { 
            if(type == 2) graph[u].pb({i + off, w});
            else graph[i].pb({u, w});
            return;
        }
        int middle = midPoint;  
        update(lp, start, end, u, w, type);    
        update(rp, start, end, u, w, type);    
    }

    void run(int s) {
        vll dp(n * 8 + 23, INF);
        min_heap<pll> q;
        q.push({0, pos[s]});
        dp[pos[s]] = 0;
        while(!q.empty()) {
            auto [cost, node] = q.top(); q.pop();
            if(dp[node] != cost) continue;
            for(auto& [nei, w] : graph[node]) {
                ll newCost = cost + w;
                if(newCost < dp[nei]) {
                    dp[nei] = newCost;
                    q.push({newCost, nei});
                }
            }
        }
        for(int i = 0; i < n; i++) {
            auto& res = dp[pos[i]];
            cout << (res == INF ? -1 : res) << (i == n - 1 ? '\n' : ' ');
        }
    }
};

class HLD {
    public:
    SGT<int> seg;
    vi id, a, tp, sz, parent;
    int ct;
    vvi graph;
    int n;
    GRAPH g;
    HLD(vvi& graph, vi& a) : seg(graph.size()), g(graph) {
        this->graph = graph;
        this->n = graph.size();
        this->a = a;
        ct = 0;
        id.rsz(n), tp.rsz(n), sz.rsz(n);
        parent.rsz(n, -1);
        dfs1();
        dfs2();
    }
     
    int dfs1(int node = 0, int par = -1) {   
        parent[node] = par;
        sz[node] = 1;   
        for(auto& nei : graph[node]) {   
            if(nei == par) continue;    
            sz[node] += dfs1(nei, node);
        }   
        return sz[node];    
    }
        
    void dfs2(int node = 0, int par = -1, int top = 0) {   
        id[node] = ct++;    
        tp[node] = top;
        int nxt = -1, max_size = -1;    
        for(auto& nei : graph[node]) {   
            if(nei == par) continue;    
            if(sz[nei] > max_size) {   
                max_size = sz[nei]; 
                nxt = nei;  
            }   
        }   
        if(nxt == -1) return;   
        dfs2(nxt, node, top);   
        for(auto& nei : graph[node]) {   
            if(nei != par && nei != nxt) dfs2(nei, node, nei);  
        }   
    }

    void update(int i, int v) {
        a[i] = v;
        seg.update(id[i], v);
    }

    int path(int node, int par) {   
        int res = 0;    
        while(node != par)  {   
            if(node == tp[node])   {   
                res += a[node];
                node = g.dp[node][0]; 
            }   else if(g.depth[tp[node]] > g.depth[par])  {   
                res += seg.queries(id[tp[node]], id[node]);    
                node = g.dp[tp[node]][0];
            }   else    {   
                res += seg.queries(id[par] + 1, id[node]);  
                break;  
            }   
        }   
        return res; 
    }

    int get_dist(int a, int b) {
        return g.dist(a, b);
    }

    int get_lca(int a, int b) {
        return g.lca(a, b);
    }
};
