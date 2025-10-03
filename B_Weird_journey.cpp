#include<bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
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
#define ull unsigned long long
#define vi vt<int>
#define vvi vt<vi>
#define vvvi vt<vvi>
#define pii pair<int, int>
#define vpii vt<pii>
#define vvpii vt<vpii>
#define vll vt<ll>  
#define vvll vt<vll>
#define pll pair<ll, ll>    
#define vpll vt<pll>
#define vvpll vt<vpll>
#define ar(x) array<int, x>
#define var(x) vt<ar(x)>
#define vvar(x) vt<var(x)>
#define al(x) array<ll, x>
#define vall(x) vt<al(x)>
#define vvall(x) vt<vall(x)>
#define vs vt<string>
#define pb push_back
#define ff first
#define ss second
#define rsz resize
#define sum(x) (ll)accumulate(all(x), 0LL)
#define srt(x) sort(all(x))
#define srtR(x) sort(allr(x))
#define srtU(x) sort(all(x)), (x).erase(unique(all(x)), (x).end())
#define rev(x) reverse(all(x))
#define MAX(a) *max_element(all(a)) 
#define MIN(a) *min_element(all(a))
#define SORTED(x) is_sorted(all(x))
#define ROTATE(a, p) rotate(begin(a), begin(a) + p, end(a))
#define i128 __int128
#define IOS ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0)
#if defined(LOCAL) && __has_include("debug.h")
  #include "debug.h"
#else
  #define debug(...)
  #define startClock
  #define endClock
  inline void printMemoryUsage() {}
#endif
template<class T> using max_heap = priority_queue<T>; template<class T> using min_heap = priority_queue<T, vector<T>, greater<T>>;
template<typename T, size_t N> istream& operator>>(istream& is, array<T, N>& arr) { for (size_t i = 0; i < N; i++) { is >> arr[i]; } return is; }
template<typename T, size_t N> istream& operator>>(istream& is, vector<array<T, N>>& vec) { for (auto &arr : vec) { is >> arr; } return is; }
template<typename T1, typename T2>  istream &operator>>(istream& in, pair<T1, T2>& input) { return in >> input.ff >> input.ss; }
template<typename T> istream &operator>>(istream &in, vector<T> &v) { for (auto &el : v) in >> el; return in; }
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
const static ll INF = 4e18 + 10;
const static int inf = 1e9 + 100;
const static int MX = 1e5 + 5;

class DSU { 
public: 
    int n, comp;  
    vi root, rank, col;  
    bool is_bipartite;  
    DSU(int n) {    
        this->n = n;    
        comp = n;
        root.rsz(n, -1), rank.rsz(n, 1), col.rsz(n, 0);
        is_bipartite = true;
    }
    
    int find(int x) {   
        if(root[x] == -1) return x; 
        int p = find(root[x]);
        col[x] ^= col[root[x]];
        return root[x] = p;
    }
    
    bool merge(int a, int b) {
        int u = find(a);
        int v = find(b);
        if (u == v) {
            if(col[a] == col[b]) {
                is_bipartite = false;
            }
            return 0;
        }
        if(rank[u] < rank[v]) {
            swap(u, v);
            swap(a, b);
        }
		comp--;
        root[v] = u;
        rank[u] += rank[v];
        if(col[a] == col[b])
            col[v] ^= 1;
        return 1;
    }
    
    bool same(int u, int v) {    
        return find(u) == find(v);
    }
    
    int get_rank(int x) {    
        return rank[find(x)];
    }
    
	vvi get_group() {
        vvi ans(n);
        for(int i = 0; i < n; i++) {
            ans[find(i)].pb(i);
        }
        sort(all(ans), [](const vi& a, const vi& b) {return a.size() > b.size();});
        while(!ans.empty() && ans.back().empty()) ans.pop_back();
        return ans;
    }
};

void solve() {
    int n, m; cin >> n >> m;
    vi degree(n);
    DSU root(n);
    vi loop_cnt(n);
    int st = 0;
    for(int i = 0; i < m; i++) {
        int u, v; cin >> u >> v;
        u--, v--;
        st = u;
        root.merge(u, v);
        if(u == v) loop_cnt[u]++;
        else {
            degree[u]++;
            degree[v]++;
        }
    }
    for(int i = 0; i < n; i++) {
        if(loop_cnt[i] || degree[i]) {
            if(!root.same(st, i)) {
                cout << 0 << '\n';
                return;
            }
        }
    }
    auto nc2 = [](ll n) -> ll {
        return n * (n - 1) / 2;
    };
    ll res = 0;
    for(auto& x : degree) {
        res += nc2(x);
    }
    ll loop = sum(loop_cnt);
    res += loop * (m - 1) - nc2(loop);
    cout << res << '\n';
}

signed main() {
    IOS;
    startClock
    int t = 1;
    //cin >> t;
    for(int i = 1; i <= t; i++) {   
        //cout << "Case #" << i << ": ";  
        solve();
    }
    endClock;
    printMemoryUsage();
    return 0;
}
