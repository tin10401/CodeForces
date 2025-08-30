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
const static ll INF = 1e18;
const static int inf = 1e9 + 100;
const static int MX = 1e5 + 5;

struct Undo_DSU {
    int n;
    using Record = ar(4);
    vi par, rank;
    stack<Record> st;

    Undo_DSU(int n) : n(n) {
        par.rsz(n);
        rank.rsz(n, 1);
        iota(par.begin(), par.end(), 0);
    }
    
    int find(int v) {
        return par[v] == v ? v : find(par[v]);
    }
    
    bool merge(int u, int v) {
        int ru = find(u), rv = find(v);
        if(ru == rv) return false;
        if(rank[ru] < rank[rv]) swap(ru, rv);
        st.push({ru, rank[ru], rv, rank[rv]});
        par[rv] = ru;
        rank[ru] += rank[rv];
        return true;
    }
    
    void rollBack() {
        if(!st.empty()) {
            Record rec = st.top();
            st.pop();
            int ru = rec[0], oldRankU = rec[1], rv = rec[2], oldRankV = rec[3];
            par[rv] = rv;
            rank[ru] = oldRankU;
            rank[rv] = oldRankV;
        }
    }

    void reset() {
        while(!st.empty()) rollBack();
    }
    
    bool same(int u, int v) {
        return find(u) == find(v);
    }
    
    int get_rank(int u) {
        return rank[find(u)];
    }
};

void solve() {
    int n, m, q; cin >> n >> m >> q;
    Undo_DSU d1(n), d2(n), d3(n);
    var(3) E0(m);
    vi W(m), C(m);
    for(int i = 0; i < m; i++) {
        auto& [u, v, id] = E0[i]; cin >> u >> v >> W[i];
        u--, v--;
        id = i;
    }
    vpii Q(q); cin >> Q;
    for(auto& [e, w] : Q) e--;
    vll ans(q);
    int test = 0;
    auto dfs = [&](auto& self, int l, int r, ll base, var(3) edges, int nv) -> void {
        test++;
        for(int i = l; i <= r; i++) {
            C[Q[i].ff] = test;
        }
        int nv2 = 0;
        auto prune = [&]() -> void {
            auto change = [&](int i) -> int {
                return C[i] == test;
            };
            d1.reset();
            d2.reset();
            d3.reset();
            sort(all(edges), [&](const ar(3)& x, const ar(3)& y) { return W[x[2]] < W[y[2]]; });
            for(auto& [u, v, i] : edges) {
                if(change(i)) {
                    d1.merge(u, v);
                }
            }
            for(auto& [u, v, i] : edges) {
                if(!change(i)) {
                    if(d1.merge(u, v)) {
                        base += W[i];
                        d2.merge(u, v);
                    }
                }
            }
            vi lab(nv, -1);
            for(int i = 0; i < nv; i++) {
                int rt = d2.find(i);
                if(lab[rt] == -1) lab[rt] = nv2++;
            }
            var(3) curr, now;
            for(auto& [u, v, id] : edges) {
                int nuu = lab[d2.find(u)];
                int nvv = lab[d2.find(v)];
                if(nuu != nvv) {
                    curr.pb({nuu, nvv, id});
                }
            }
            for(auto& [u, v, i] : curr) {
                if(!change(i)) {
                    if(d3.merge(u, v)) now.pb({u, v, i});
                }
            }
            for(auto& [u, v, i] : curr) {
                if(change(i)) now.pb({u, v, i});
            }
            swap(now, edges);
        };
        auto mst = [&]() -> ll {
            ll res = 0;
            sort(all(edges), [&](const ar(3)& x, const ar(3)& y) { return W[x[2]] < W[y[2]]; });
            d3.reset();
            for(auto& [u, v, id] : edges) {
                if(d3.merge(u, v)) {
                    res += W[id];
                }
            }
            return res;
        };
        prune();
        if(l == r) {
            auto& [e, w] = Q[l];
            W[e] = w;
            ans[l] = base + mst(); 
            return;
        }
        int mid = (l + r) >> 1;
        self(self, l, mid, base, edges, nv2);
        self(self, mid + 1, r, base, edges, nv2);
    };
    dfs(dfs, 0, q - 1, 0, E0, n);
    for(auto& x : ans) cout << x << '\n';
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
