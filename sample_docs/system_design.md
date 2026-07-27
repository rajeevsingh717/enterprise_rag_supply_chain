---
version: "1.0"
title: System Design Notes
---

# System Design

## Networking

### Subnets

Subnets divide a larger network into smaller, more manageable routable
segments. Each subnet is defined by a CIDR block that specifies its address
range and the number of usable host addresses within it. Routers forward
packets between subnets based on entries in a routing table, examining the
destination address of each packet to decide the next hop. Network
administrators size subnets according to the expected number of hosts,
leaving room for growth without wasting address space. A well-planned
subnet layout reduces broadcast traffic significantly, since broadcasts
stay contained within a single subnet instead of flooding the entire
network. Overlapping subnet ranges are a common source of routing
conflicts, especially when merging networks after an acquisition or a
cloud migration. Cloud providers let you define custom subnet ranges per
availability zone, which is useful for isolating workloads and controlling
blast radius during an incident. Subnet masks determine which bits of an
address identify the network portion versus the host portion, and getting
this wrong is one of the most common networking misconfigurations.

### Load Balancing

Load balancers distribute incoming traffic across a pool of backend
servers so that no single server becomes a bottleneck. A common algorithm
is round robin, which cycles through the pool in order, though it ignores
the current load on each server. Health checks periodically probe each
backend to confirm it is still responding correctly, automatically
removing unhealthy nodes from rotation until they recover. Sticky sessions
route a given client to the same backend for the duration of a session,
which simplifies stateful applications but can create hot spots. Layer 4
load balancers operate on raw TCP or UDP connections and are extremely
fast, while layer 7 load balancers inspect HTTP headers and can route
based on path or hostname. Autoscaling groups often sit behind a load
balancer, adding or removing backend instances as traffic rises and falls
throughout the day.

## Databases

### Indexing

Database indexes speed up query lookups by avoiding a full scan of every
row in a table. A B-tree index keeps keys sorted so that range queries and
equality lookups are both efficient, which is why it is the default index
type in most relational databases. The query planner chooses which index
to use, if any, based on estimated selectivity and the shape of the query.
Composite indexes cover queries that filter or sort on multiple columns at
once, but the column order in the index definition matters a great deal.
Index maintenance adds overhead to every insert, update, and delete
statement, since the index structure has to be kept in sync with the
underlying table. Covering indexes let the database satisfy a query
entirely from the index itself, without a second lookup against the table.
Poorly chosen indexes bloat storage and slow down writes without
meaningfully improving read performance, so database administrators
regularly monitor index usage and prune the ones that never get used.

### Replication

Replication keeps copies of a database on multiple nodes so that a single
node failure does not take down the whole system. In a primary-replica
setup, all writes go to the primary node and are then streamed to the
replicas, which typically serve read-only traffic. Synchronous replication
waits for a replica to confirm a write before acknowledging it to the
client, trading latency for durability. Asynchronous replication
acknowledges the write immediately and streams it to replicas afterward,
which is faster but risks losing recent writes if the primary fails.
Failover promotes a replica to primary when the original primary becomes
unreachable, and the mechanics of that promotion are one of the trickiest
parts of running a distributed database in production. Replication lag,
the delay between a write landing on the primary and appearing on a
replica, is a metric every on-call engineer eventually has to reason
about.
