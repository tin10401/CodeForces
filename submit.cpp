// Author : Tin Le
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
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) begin(x), end(x)
#define allr(x) rbegin(x), rend(x)
#define ub upper_bound
#define lb lower_bound
#define db double
#define ll unsigned long long
#define int long long
#define vi vector<int>
#define pii pair<int, int>
#define vpii vector<pair<int, int>>
#define vvi vector<vi>
#define vd vector<db>
#define ar(x) array<int, x>
#define mset(m, v) memset(m, v, sizeof(m))
#define pb push_back
#define f first
#define s second
#define rsz resize
#define sum(x) accumulate(all(x), 0LL)
#define srt(x) sort(all(x))
#define srtR(x) sort(allr(x))
#define srtU(x) sort(all(x)), (x).erase(unique(all(x)), (x).end())
#define rev(x) reverse(all(x))
#define IOS ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0)
#ifdef LOCAL
#define startClock clock_t tStart = clock();
#define endClock cout << fixed << setprecision(10) << "\nTime Taken: " << (double)(clock() - tStart) / CLOCKS_PER_SEC << endl;
#else
#define startClock
#define endClock
#endif
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

const static int INF = 1LL << 61;
const static int MX = 2e5 + 5;
const static int MOD = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
constexpr int pct(int x) { return __builtin_popcount(x); }
const vvi dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
constexpr int modExpo(int base, int exp, int mod) { int res = 1; while(exp) {
    if(exp & 1) res = (res * base) % mod; base = (base * base) % mod; exp >>=
        1; } return res; }

int sieve[MX];
void generate()
{
    for(int i = 1; i < MX; i++) sieve[i] = i;
    for(int i = 2; i < MX; i += 2) sieve[i] = 2;
    for(int i = 3; i * i < MX; i += 2)
    {
        if(sieve[i] == i)
        {
            for(int j = i; j * i < MX; j += 2)
            {
                sieve[i * j] = min(sieve[i * j], i);
            }
        }
    }
}


class SGT
{
    public:
    int n;
    vector<ar(3)> root;
    SGT(vi& arr)
    {
        n = arr.size();
        root.rsz(n * 4);
        build(0, 0, n - 1, arr);
    }

    void build(int i, int left, int right, vi& arr)
    {
        if(left == right)
        {
            root[i][0] = arr[left];
            root[i][1] = sieve[arr[left]];
            root[i][2] = root[i][1] == 1;
            return;
        }
        int middle = left + (right - left) / 2;
        build(i * 2 + 1, left, middle, arr);
        build(i * 2 + 2, middle + 1, right, arr);
        root[i] = merge(root[i * 2 + 1], root[i * 2 + 2]);
    }

    ar(3) merge(ar(3)& a, ar(3)& b)
    {
        ar(3) res = {};
        res[1] = max(a[1], b[1]);
        res[2] = a[2] & b[2];
        return res;
    }

    void update(int start, int end)
    {
        update(0, 0, n - 1, start, end);
    }

    void update(int i, int left, int right, int start, int end)
    {
        if(left > end || start > right || root[i][2]) return;
        if(left == right && left >= start && right <= end)
        {
            root[i][0] /= sieve[root[i][0]];
            root[i][1] = sieve[root[i][0]];
            if(root[i][1] == 1) root[i][2] = true;
            return;
        }
        int middle = left + (right - left) / 2;
        update(i * 2 + 1, left, middle, start, end);
        update(i * 2 + 2, middle + 1, right, start, end);
        root[i] = merge(root[i * 2 + 1], root[i * 2 + 2]);
    }

    int get(int start, int end)
    {
        return get(0, 0, n - 1, start, end);
    }

    int get(int i, int left, int right, int start, int end)
    {
        if(left > end || start > right) return 0;
        if(left >= start && right <= end) return root[i][1];
        int middle = left + (right - left) / 2;
        return max(get(i * 2 + 1, left, middle, start, end), get(i * 2 + 2, middle + 1, right, start, end));
    }
};



void solve()
{
    int n, q; cin >> n >> q;
    vi arr(n);
    for(auto& it : arr) cin >> it;
    SGT root(arr);
    while(q--)
    {
        int type, a, b; cin >> type >> a >> b;
        if(type) cout << root.get(--a, --b) << " ";
        else root.update(--a, --b);
    }
    cout << endl;
}

signed main()
{
    IOS;
    startClock

    generate(); 
    int t = 1;
     cin >> t;
    while(t--) solve();

    endClock
    return 0;
}

