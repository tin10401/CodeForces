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
#define vs vector<string>
#define vb vector<bool>
#define us unordered_set
#define um unordered_map
#define vvi vector<vi>
#define vd vector<db>
#define ar(x) array<int, x>
#define var(x) vector<ar(x)>
#define pq priority_queue
#define mset(m, v) memset(m, v, sizeof(m))
#define pb push_back
#define umii um<int, int, custom>
#define usi us<int, custom>
#define ff first
#define ss second
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
const static int MX = 2e6 + 5;
const static int MOD = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
constexpr int pct(int x) { return __builtin_popcount(x); }
const vvi dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
constexpr int modExpo(int base, int exp, int mod) { int res = 1; while(exp) { if(exp & 1) res = (res * base) % mod; base = (base * base) % mod; exp >>= 1; } return res; }
struct custom {
    static uint64_t splitmix64(uint64_t x) { x += 0x9e3779b97f4a7c15; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9; x = (x ^ (x >> 27)) * 0x94d049bb133111eb; return x ^ (x >> 31); }
    size_t operator()(uint64_t x) const { static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count(); return splitmix64(x + FIXED_RANDOM); }
};

struct Node 
{   
    int sm, mx, val, lz;    
    Node() : mx(-1), sm(0), val(0), lz(-1) {} 
    Node(int v) : val(v), sm(v), lz(-1), mx(v) {};
};  
    
static Node root[MX * 4];
    
class SGT   
{   
    public: 
    int n;  
    SGT(vi& arr)    
    {   
        n = arr.size(); 
        build(0, 0, n - 1, arr);    
    }   
    
    void build(int i, int left, int right, vi& arr) 
    {   
        if(left == right)   
        {   
            root[i] = Node(arr[left]);  
            return; 
        }   
        int middle = left + (right - left) / 2; 
        build(i * 2 + 1, left, middle, arr);    
        build(i * 2 + 2, middle + 1, right, arr);   
        root[i] = merge(root[i * 2 + 1], root[i * 2 + 2]);  
    }   
    
    Node merge(Node left, Node right)   
    {   
        Node res;   
        res.sm = left.sm + right.sm;    
        res.mx = max(left.mx, right.mx);    
        res.val = left.val == right.val && left.val > 0 ? left.val : -1;    
        return res; 
    }   
    
    void update(int start, int end, int x, bool g)  
    {   
        update(0, 0, n - 1, start, end, x, g);  
    }   
    
    void update(int i, int left, int right, int start, int end, int x, int g)    
    {   
        push(i, left, right); 
        if(left > end || start > right) return;
        if(left >= start && right <= end && root[i].val > 0)    
        {   
            root[i].lz = (g ? gcd(root[i].val, x) : x);   
            push(i, left, right);
            return; 
        }   
        int middle = left + (right - left) / 2; 
        update(i * 2 + 1, left, middle, start, end, x, g);  
        update(i * 2 + 2, middle + 1, right, start, end, x, g); 
        root[i] = merge(root[i * 2 + 1], root[i * 2 + 2]);  
    }   
    
    void push(int i, int left, int right)   
    {   
        int& x = root[i].lz;
        if(x == -1) return;
        root[i].sm = (right - left + 1) * x;
        root[i].mx = root[i].val = x;
        if(left != right)   
        {   
            root[i * 2 + 1].lz = x;
            root[i * 2 + 2].lz = x;
        }   
        x = -1;
    }   
    
    Node get(int start, int end)    
    {   
        return get(0, 0, n - 1, start, end);    
    }   
    
    Node get(int i, int left, int right, int start, int end)    
    {   
        push(i, left, right);
        if(left > end || start > right) return Node();  
        if(left >= start && right <= end) return root[i];   
        int middle = left + (right - left) / 2; 
        return merge(get(i * 2 + 1, left, middle, start, end), get(i * 2 + 2, middle + 1, right, start, end));
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
        int type; cin >> type;  
        if(type == 1 || type == 2)   
        {   
            int l, r, x; cin >> l >> r >> x;    
            l--, r--;   
            root.update(l, r, x, type == 2);    
        }   
        else    
        {   
            int l, r; cin >> l >> r;    
            l--, r--;
            if(type == 3) cout << root.get(l, r).mx << endl;    
            else cout << root.get(l, r).sm << endl; 
        }   
    }


}

signed main()
{
    IOS;
    startClock

    int t = 1;
    // cin >> t;
    while(t--) solve();

    endClock
    return 0;
}

