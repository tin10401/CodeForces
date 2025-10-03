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
const static int MX = 3e5 + 5;

bitset<MX> primeBits;
int phi[MX], spf[MX], mu[MX];
ll lcm_sum[MX], gcd_sum[MX];
vi primes, DIV[MX];

void nt_processing() {  
	primeBits.set(2);   
    for(int i = 3; i < MX; i += 2) primeBits.set(i);
    for(int i = 2; i * i < MX; i += (i == 2 ? 1 : 2)) {    
        if(primeBits[i]) {  
            for(int j = i; j * i < MX; j += 2) {    primeBits.reset(i * j); }
        }
    }
    iota(all(phi), 0);
    mu[1] = 1;
    for(int i = 2; i < MX; i++) {
        if(spf[i] == 0) {
            primes.pb(i);
            for(int j = i; j < MX; j += i) {    
                if(spf[j] == 0) spf[j] = i; 
                phi[j] -= phi[j] / i;
            }
        }
        int p = spf[i];
        int m = i / p;
        mu[i] = m % p == 0 ? 0 : -mu[m];
    }
    for(int d = 1; d < MX; d++) {
        for(int j = d; j < MX; j += d) {
            gcd_sum[j] += phi[j / d] * (ll)d; // for odd sum  gcd([1, 3, 5, ..], n), do (gcd_sum[n] + n) / 2 because of symmetry gcd(n, 1) == gcd(n, n - 1), + n because there are no pair matching gcd(n, n)
        }
    }
    for(int d = 1; d < MX; ++d) {
        ll term = (ll)d * phi[d];
        for(int n = d; n < MX; n += d) {
            lcm_sum[n] += term;
        }
    }
    for(int n = 1; n < MX; ++n) {
        lcm_sum[n] = (lcm_sum[n] + 1) * (ll)n / 2;
    }
} static const bool _nt_init = []() { nt_processing(); return true; }();

struct EulerianPath {
    int nodes, edges;
    bool directed;
    vvpii graph;
    vi deg, indeg, outdeg;
    vt<bool> used;
    vi ans_edges, ans_nodes;

    EulerianPath(int _nodes, bool _directed = false)
      : nodes(_nodes), edges(0), directed(_directed), graph(_nodes) {
        if(directed) indeg.assign(nodes,0), outdeg.assign(nodes,0);
        else deg.assign(nodes,0);
    }

    void add_edge(int u, int v, int id) {
        graph[u].emplace_back(v, id);
        edges++;
        if(directed) {
            outdeg[u]++;
            indeg[v]++;
        } else {
            graph[v].emplace_back(u, id);
            deg[u]++;
            deg[v]++;
        }
    }

    int find_start() const {
        int start = -1;
        if(!directed) {
            int odd = 0;
            for(int i = 0; i < nodes; i++) {
                if(deg[i] & 1) {
                    odd++;
                    start = i;
                }
                if(start < 0 && deg[i] > 0) start = i;
            }
            if(start < 0) return 0;
            if(odd != 0 && odd != 2) return -1;
        } else {
            int plus1 = 0, minus1 = 0;
            for(int i = 0; i < nodes; i++) {
                int d = outdeg[i] - indeg[i];
                if(d == 1) { plus1++; start = i; }
                else if(d == -1) minus1++;
                else if(d != 0) return -1;
                if(start < 0 && outdeg[i] > 0) start = i;
            }
            if(start < 0) return 0;
            if(!((plus1 == 1 && minus1 == 1) || (plus1 == 0 && minus1 == 0))) return -1;
        }
        return start;
    }

    void dfs(int u) {
		if(used.empty()) {
			used.rsz(edges);
		}
        while(!graph[u].empty()) {
            auto [v, id] = graph[u].back();
            graph[u].pop_back();
			while((int)used.size() <= id) used.pb(0);
            if(used[id]) continue;
            used[id] = true;
            dfs(v);
            ans_edges.pb(id);
        }
        ans_nodes.pb(u);
    }

    pair<vi, vi> get_path() {
        used.rsz(edges);
        dfs(0);
        rev(ans_nodes);
        rev(ans_edges);
        return {ans_nodes, ans_edges};
    }
};

void solve() {
    int n; cin >> n;
    vi ans;
    auto f = [&](ll k) -> ll {
        ll res = k * (k - 1) / 2 + k - (k % 2 == 0 ? (k - 2) / 2 : 0) + 1;
        return res;
    };
    int left = 1, right = 10000, k = -1;
    while(left <= right) {
        int middle = (left + right) >> 1;
        if(f(middle) >= n) k = middle, right = middle - 1;
        else left = middle + 1;
    }
    debug(k, f(k));
    EulerianPath graph(k, false); 
    int m = 0;
    for(int i = 0; i < k; i++) {
        for(int j = i; j < k; j++) {
            if(k % 2 == 0 && i % 2 == 1 && j == i + 1) continue;
            graph.add_edge(i, j, m++);
        }
    }
    auto [nodes, edges] = graph.get_path();
    swap(ans, nodes);
    assert((int)ans.size() >= n);
    ans.rsz(n);
    for(auto& x : ans) cout << primes[x] << ' ';
    cout << '\n';
}

signed main() {
    IOS;
    startClock
    int t = 1;
    cin >> t;
    for(int i = 1; i <= t; i++) {   
        //cout << "Case #" << i << ": ";  
        solve();
    }
    endClock;
    printMemoryUsage();
    return 0;
}
