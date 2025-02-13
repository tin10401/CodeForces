struct Point {
    ld x, y;
    Point() : x(0), y(0) {}
    Point(ld x, ld y) : x(x), y(y) {}
    double distance(const Point &other) const {
        return std::sqrt((x - other.x) * (x - other.x) +
                         (y - other.y) * (y - other.y));
    }
    friend std::ostream& operator<<(std::ostream& os, const Point& p) {
        os << "(" << p.x << ", " << p.y << ")";
        return os;
    }

    ld slope_to(const Point &other) const {
        if (std::fabs(other.x - x) < eps) return std::numeric_limits<ld>::infinity();
        return (other.y - y) / (other.x - x);
    }
};

bool collinear(const Point &a, const Point &b, const Point &c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) == 0;
}

class CHT {
    public:
    int is_mx;
    vll m_slopes, b_intercepts;
    CHT(int is_mx) : is_mx(is_mx) {
        add_line(0, 0);
    }

    db cross(int i, int j, int k) {
        db A = (db)(1.00 * m_slopes[j] - m_slopes[i]) * (b_intercepts[k] - b_intercepts[i]);
        db B = (db)(1.00 * m_slopes[k] - m_slopes[i]) * (b_intercepts[j] - b_intercepts[i]);
        return is_mx ? A < B : A >= B;
    }

    void add(ll a, ll b) {
        if(is_mx) add_line(a, -b);
        else add_line(-a, b);
    }

    ll queries(ll x) {
        return is_mx ? -get(x) : get(x);
    }

    void add_line(ll slope, ll intercept) {
        m_slopes.push_back(slope);
        b_intercepts.push_back(intercept);
        while(m_slopes.size() >= 3 && cross(m_slopes.size() - 3, m_slopes.size() - 2, m_slopes.size() - 1)) {
            m_slopes.erase(m_slopes.end() - 2);
            b_intercepts.erase(b_intercepts.end() - 2);
        }
    }

    ll get(ll x) {
        if(m_slopes.empty()) return INF;
        int l = 0, r = m_slopes.size() - 1;
        while(l < r) {
            int mid = l + (r - l) / 2;
            ll f1 = m_slopes[mid] * x + b_intercepts[mid];
            ll f2 = m_slopes[mid + 1] * x + b_intercepts[mid + 1];
            if(f1 > f2) l = mid + 1;
            else r = mid;
        }
        return m_slopes[l] * x + b_intercepts[l];
    }
};

struct Line {
    mutable ll m, c, p;
    bool isQuery;
    bool operator<(const Line& o) const {
        if(o.isQuery)
            return p < o.p;
        return m < o.m;
    }
};

vvi rotate90(const vvi &matrix) {

struct CHT : multiset<Line> {
    const ll inf = INF;
    int is_mx;
    ll div(ll a, ll b) {
        return a / b - ((a ^ b) < 0 && a % b); }
    bool isect(iterator x, iterator y) {
        if (y == end()) { x->p = inf; return false; }
        if (x->m == y->m) x->p = x->c > y->c ? inf : -inf;
        else x->p = div(y->c - x->c, x->m - y->m);
        return x->p >= y->p;
    }
    void add(ll m, ll c) {
        auto z = insert({m, c, 0, 0}), y = z++, x = y;
        while (isect(y, z)) z = erase(z);
        if (x != begin() && isect(--x, y)) isect(x, y = erase(y));
        while ((y = x) != begin() && (--x)->p >= y->p)
            isect(x, erase(y));
    }
    ll query(ll x) {
        if(empty()) return inf;
        Line q; q.p = x, q.isQuery = 1;
        auto l = *lower_bound(q);
        return l.m * x + l.c;
    }
    // min will return -ans;
    // max will return ans;
    // max_normall is add(i, -dp)
    // min_normal is add(-i, dp)
};