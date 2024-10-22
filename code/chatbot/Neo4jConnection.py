from neo4j import GraphDatabase

class Neo4jConnection:
    def __init__(self, uri, user, password, db=None):
        self.__uri = uri
        self.__user = user
        self.__password = password
        self.__db = db
        self.is_alive = False
        try:
            with GraphDatabase.driver(self.__uri, auth=(self.__user, self.__password)) as driver:
                driver.verify_connectivity()    
        except Exception as e:
            print("Failed to create the driver:", e)
        else:
            self.is_alive = True

    def query(self, query, **parameters):
        response = None
        success = False
        try:
            with GraphDatabase.driver(self.__uri, auth=(self.__user, self.__password)) as driver:
                with driver.session(database=self.__db) as session:
                    response = list(session.run(query, **parameters))
        except Exception as e:
            print("Query failed:", e)
        else:
            success = True
        return response, success
    
    def bulk_query(self, queries, **parameters):
        response = None
        try:
            with GraphDatabase.driver(self.__uri, auth=(self.__user, self.__password)) as driver:
                with driver.session(database=self.__db) as session:
                    response = []
                    # Use transaction to execute all queries within one transaction block
                    with session.begin_transaction() as txn:
                        for query in queries:
                            result = txn.run(query, parameters)
                            response.append(list(result))
                        # Commit the transaction after all queries are run
                        txn.commit()
        except Exception as e:
            print("Query failed:", e)
        return response

    
    def run_query(self, cypher):
        query_result, success = self.query(cypher)
        if success:
            return [dict(record) for record in query_result]

if __name__ == "__main__":
    import os
    # Test the connection
    uri = os.getenv("NEO4J_URI")
    AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD"))
    driver = Neo4jConnection(uri, *AUTH, "neo4j")
    