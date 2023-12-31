// ResultsScreen.js

import React, { useState, useEffect } from 'react';
import { FlatList, View, Text, RefreshControl } from 'react-native';
import styles from '../styles/ResultsStyle';

const ResultsScreen = () => {
  const [data, setData] = useState([]);
  const [refreshing, setRefreshing] = useState(false);

  const fetchData = async () => {
    try {
      const response = await fetch('https://tgryl.pl/quiz/results?last=20');
      const result = await response.json();
      setData(result);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    fetchData().finally(() => setRefreshing(false));
  };

  useEffect(() => {
    fetchData();
  }, []);

  return (
    <FlatList
      data={data}
      keyExtractor={(item, index) => index.toString()}
      renderItem={({ item, index }) => (
        <View style={[styles.row, index % 2 === 0 ? styles.evenRow : styles.oddRow]}>
          <Text style={styles.cell}>{item.nick}</Text>
          <Text style={styles.cell}>{item.score}/{item.total}</Text>
          <Text style={styles.cell}>{item.type}</Text>
          <Text style={styles.cell}>{item.createdOn}</Text>
        </View>
      )}
      ListHeaderComponent={() => (
        <View style={[styles.row, styles.header]}>
          <Text style={styles.headerCell}>Nick</Text>
          <Text style={styles.headerCell}>Point</Text>
          <Text style={styles.headerCell}>Type</Text>
          <Text style={styles.headerCell}>Date</Text>
        </View>
      )}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
    />
  );
};

export default ResultsScreen;