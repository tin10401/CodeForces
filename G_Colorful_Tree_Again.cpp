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

template<typename T, typename lazy_type = ll>
struct lazy_seg {
    int n, n0, h;
    vt<T> tree;
    vi seglen;

    lazy_seg(int n_) : n(n_) , n0(1) , h(0) {
        while(n0 < n) {
            n0 <<= 1;
            ++h;
        }
        tree.assign(2 * n0, T());
        seglen.assign(2 * n0, 0);
        for (int i = n0; i < 2 * n0; ++i) {
            seglen[i] = 1;
        }
        for (int i = n0 - 1; i > 0; --i) {
            seglen[i] = seglen[i * 2] + seglen[i * 2 + 1];
        }
    }

    void apply_node(int p, lazy_type v) {
        tree[p].apply(v, seglen[p]);
    }

    void pull(int p) {
        tree[p] = tree[2 * p] + tree[2 * p + 1];
    }

    void push(int p) {
        if(tree[p].have_lazy()) {
            apply_node(2 * p, tree[p].lazy);
            apply_node(2 * p + 1, tree[p].lazy);
            tree[p].reset_lazy();
        }
    }

    void push_to(int p) {
        for (int i = h; i > 0; --i) {
            push(p >> i);
        }
    }

    void update_range(int l, int r, lazy_type v) {
        if(l > r) return;
        l = max(0, l);
        r = min(r, n - 1);
        int L = l + n0;
        int R = r + n0;
        push_to(L);
        push_to(R);
        int l0 = L, r0 = R + 1;
        while(l0 < r0) {
            if(l0 & 1) apply_node(l0++, v);
            if(r0 & 1) apply_node(--r0, v);
            l0 >>= 1;
            r0 >>= 1;
        }
        for(int i = 1; i <= h; ++i) {
            if(((L >> i) << i) != L) {
                pull(L >> i);
            }
            if((((R + 1) >> i) << i) != (R + 1)) {
                pull(R >> i);
            }
        }
    }

    void update_at(int p, T v) {
        if(p < 0 || p >= n) return;
        int pos = p + n0;
        push_to(pos);
        tree[pos] = v;
        for(pos >>= 1; pos > 0; pos >>= 1) {
            pull(pos);
        }
    }

    T queries_at(int p) {
        if(p < 0 || p >= n) return T();
        int pos = p + n0;
        push_to(pos);
        return tree[pos];
    }

    T queries_range(int l, int r) {
        if(l > r) return T();
        l = max(0, l);
        r = min(r, n - 1);
        int L = l + n0;
        int R = r + n0;
        push_to(L);
        push_to(R);
        T resL;
        T resR;
        int l0 = L;
        int r0 = R + 1;
        while(l0 < r0) {
            if(l0 & 1) resL = resL + tree[l0++];
            if(r0 & 1) resR = tree[--r0] + resR;
            l0 >>= 1;
            r0 >>= 1;
        }
        return (resL + resR);
    }

    T get() {
        return queries_range(0, n - 1);
    }

    template<typename Pred>
        int max_right(int l, Pred P) {
            if(l < 0) l = 0;
            if(l >= n) return n;
            T sm;
            int idx = l + n0;
            push_to(idx);
            int tmp = idx;
            do {
                while((tmp & 1) == 0) tmp >>= 1;
                T cand = sm + tree[tmp];
                if(!P(cand)) {
                    while(tmp < n0) {
                        push(tmp);
                        tmp <<= 1;
                        T cand2 = sm + tree[tmp];
                        if(P(cand2)) {
                            sm = cand2;
                            tmp++;
                        }
                    }
                    return tmp - n0 - 1;
                }
                sm = sm + tree[tmp];
                tmp++;
            } while((tmp & -tmp) != tmp);
            return n - 1;
        }

    template<typename Pred>
        int min_left(int r, Pred P) {
            if(r < 0) return 0;
            if(r >= n) r = n - 1;
            T sm;
            int idx = r + n0 + 1;
            push_to(idx - 1);
            do {
                idx--;
                while(idx > 1 && (idx & 1)) idx >>= 1;
                T cand = tree[idx] + sm;
                if(!P(cand)) {
                    while(idx < n0) {
                        push(idx);
                        idx = idx * 2 + 1;
                        T cand2 = tree[idx] + sm;
                        if(P(cand2)) {
                            sm = cand2;
                            idx--;
                        }
                    }
                    return idx + 1 - n0;
                }
                sm = tree[idx] + sm;
            } while((idx & -idx) != idx);
            return 0;
        }
};

struct info {
    const static ll lazy_value = 0;
    ll mx, cnt;
    ll lazy;
    info(ll v = -INF, int _cnt = inf) : mx(v), cnt(_cnt), lazy(lazy_value) { }

    int have_lazy() {
        return !(lazy == lazy_value);
    }

    void reset_lazy() {
        lazy = lazy_value;
    }

    void apply(ll v, int len) {
        cnt += v;
        lazy += v;
    }

    friend info operator+(const info& a, const info& b) { // careful about lazy_copy
        info res;
        res.cnt = min(a.cnt, b.cnt);
        res.mx = max(a.cnt == res.cnt ? a.mx : 0, b.cnt == res.cnt ? b.mx : 0);
        return res;
    }

    ll get() {
        return cnt == 0 ? mx : 0;
    }
};

void solve() {
    int n, q; cin >> n >> q;
    vvpii graph(n + 1);
    vi ok(n + 1);
    vvar(3) color(n + 1);
    for(int i = 1; i < n; i++) {
        int u, v, w, c; cin >> u >> v >> w >> c;
        graph[u].pb({v, c});
        graph[v].pb({u, c});
        color[c].pb({u, v, w});
    }
    vll weight(n + 1);
    vi degree(n + 1);
    for(int c = 1; c <= n; c++) {
        ll W = 0; 
        vi a;
        for(auto& [u, v, w] : color[c]) {
            if(!degree[u]++) {
                a.pb(u);
            }
            if(!degree[v]++) {
                a.pb(v);
            }
            W += w;
        }
        int one = 0, mx = 0;
        for(auto& x : a) {
            if(degree[x] == 1) one++;
            mx = max(mx, degree[x]);
            degree[x] = 0;
        }
        if(mx <= 2 && one == 2) {
            weight[c] = W;
            ok[c] = true;
        }
    }
    vi id(n + 1);
    vi L(n + 1, inf), R(n + 1);
    int timer = 0;
    vi p_index(n + 1);
    auto dfs = [&](auto& dfs, int u = 1, int par = 0) -> void {
        for(auto& [v, c] : graph[u]) {
            if(v == par) continue;
            if(ok[c]) {
                if(!id[c]) { 
                    id[c] = ++timer;
                    L[u] = min(L[u], id[c]);
                    R[u] = max(R[u], id[c]);
                }
                p_index[v] = id[c];
            }
        }
        for(auto& [v, c] : graph[u]) {
            if(v == par) continue;
            dfs(dfs, v, u);
        }
    };
    dfs(dfs);
    lazy_seg<info> root(n + 1);
    for(int i = 1; i <= n; i++) {
        if(ok[i]) {
            root.update_at(id[i], info(weight[i], 0));
        }
    }
    while(q--) {
        int p, x; cin >> p >> x;
        int delta = p == 0 ? 1 : -1;
        if(L[x] <= R[x]) root.update_range(L[x], R[x], delta);
        if(p_index[x]) root.update_range(p_index[x], p_index[x], delta);
        cout << root.get().get() << '\n';
    }
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
