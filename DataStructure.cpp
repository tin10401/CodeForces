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
#define iterator int i, int left, int right

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
        T val;
        TreapNode* left;
        TreapNode* right;
        
        TreapNode(char val) : f(0), val(val), pri(rand()), size(1), left(nullptr), right(nullptr) {}
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

    void insert(T ch) { 
        merge(root, root, new TreapNode(ch));
    }
    
    void split_and_reverse(int l, int r) { 
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
    
    T get(int k) {  
        TreapNode* A, *B, *C;   
        split(root, A, B, k - 1);
        split(B, B, C, 1);  
        T ans = B->val;
        merge(root, A, B);  
        merge(root, root, C);
        return ans;
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
    int n;  
    vt<T> root;    
    FW(int n) { 
        this->n = n;    
        root.rsz(n + 1);
    }
    
    void update(int id, T val) {  
        while(id <= n) {    
            root[id] += val;    
            id += (id & -id);
        }
    }
    
    T get(int id) {   
        T res = 0;    
        while(id > 0) { 
            res += root[id];    
            id -= (id & -id);
        }
        return res;
    }
    
    T queries(int left, int right) {  
        return get(right) - get(left - 1);
    }
	
	void reset() {
		root.assign(n, 0);
	}
};

template<class T>   
class SGT { 
    public: 
    int n;  
    vt<T> root;
	vi lazy;
    T DEFAULT;
    SGT(vi& arr) {    
        n = arr.size(); 
        DEFAULT = INF;
        root.rsz(n * 4);    
        lazy.rsz(n * 4);
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
//            int middle = midPoint;  
//            root[lc] = (ll)(middle - left + 1) * lazy[i];
//            root[rc] = (ll)(right - middle) * lazy[i];
        }
        lazy[i] = 0;
    }

	T queries(int id) {
		return queries(entireTree, id);
	}
	
	T queries(iterator, int id) {
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
    
    T queries(iterator, int start, int end) {   
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
    
    void print(iterator) {  
        pushDown;
        if(left == right) { 
            cout << root[i] << ' ';
            return;
        }
        int middle = midPoint;  
        print(lp);  print(rp);
    }

};

// PERSISTENT SEGTREE
int T[MX * MK * 4], root[MX * MK * 4], ptr, n, m; 
pii child[MX * MK * 4];
void update(int curr, int prev, int id, int left, int right) {  
    root[curr] = root[prev];    
    child[curr] = child[prev];
    if(left == right) { 
        root[curr]++;
        return;
    }
    int middle = midPoint;
    if(id <= middle) {  
        child[curr].ff = ++ptr; 
        update(child[curr].ff, child[prev].ff, id, left, middle);
    }
    else {  
        child[curr].ss = ++ptr; 
        update(child[curr].ss, child[prev].ss, id, middle + 1, right);
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
        int left = 0, right = 0;    // modify to 0 as needed "left = 0"
        for(auto& [l, r, id] : Q) { 
            while(left < l) {  
                modify(a[left++], -1);
            }
            while(left > l) {   
                modify(a[--left], 1);
            }
            while(right > r) {   
                modify(a[--right], -1);
            }
            while(right <= r) { 
                modify(a[right++], 1);
            }
            res[id] = ans;
        }
        return res;
    }
};

class SparseTable { 
    public: 
    int n;  
    vvll dp; 
    SparseTable(vvll& dp) {  
        n = dp.size();  
        this->dp = dp;
        init();
    }
    
    void init() {   
        for(int j = 1; j < MK; j++) {    
            for(int i = 0; i + (1LL << j) <= n; i++) {    
				//dp[i][j] = gcd(dp[i][j - 1], dp[i + (1LL << (j - 1))][j - 1]);
                //dp[i][j] = min(dp[i][j - 1], dp[i + (1LL << (j - 1))][j - 1]);
                //dp[i][j] = max(dp[i][j - 1], dp[i + (1LL << (j - 1))][j - 1]);
            }
        }
    }
    
    ll queries(int left, int right) {  
        int j = log2(right - left + 1);
        return gcd(dp[left][j], dp[right - (1LL << j) + 1][j]);
    }
};

class TWO_DIMENSIONAL_RANGE_QUERY {   
    public: 
    vvi prefix, grid; 
    int n, m;
    TWO_DIMENSIONAL_RANGE_QUERY(vvi& grid) {  
        n = grid.size(), m = grid[0].size();
        this->grid = grid;
        prefix.assign(n + 1, vi(m + 1));  
        init();
    }
    
    ll getSum(int r1, int r2, int c1, int c2) {   
        int bottomRight = prefix[r2][c2];   
        int topLeft = prefix[r1 - 1][c1 - 1];
        int topRight = prefix[r1 - 1][c2];  
        int bottomLeft = prefix[r2][c1 - 1];
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


