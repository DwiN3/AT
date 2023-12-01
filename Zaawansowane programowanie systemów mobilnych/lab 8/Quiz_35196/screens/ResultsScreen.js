// ResultsScreen.js

import React, { useState } from 'react';
import { FlatList, View, Text, RefreshControl } from 'react-native';
import styles from '../styles/ResultsStyle';
import { userScores } from '../data/UserScores';

const ResultsScreen = () => {
  const [data, setData] = useState(userScores);
  const [refreshing, setRefreshing] = useState(false);

  const onRefresh = () => {
    setRefreshing(true);
    setTimeout(() => {
      setData(userScores);
      setRefreshing(false);
    }, 1000);
  };

  return (
    <FlatList
      data={data}
      keyExtractor={(item, index) => index.toString()}
      renderItem={({ item, index }) => (
        <View style={[styles.row, index % 2 === 0 ? styles.evenRow : styles.oddRow]}>
          <Text style={styles.cell}>{item.nick}</Text>
          <Text style={styles.cell}>{item.score}/{item.total}</Text>
          <Text style={styles.cell}>{item.type}</Text>
          <Text style={styles.cell}>{item.date}</Text>
        </View>
      )}
      ListHeaderComponent={() => (
        <View style={[styles.row, styles.header]}>
          <Text style={styles.cell}>Nick</Text>
          <Text style={styles.cell}>Point</Text>
          <Text style={styles.cell}>Type</Text>
          <Text style={styles.cell}>Date</Text>
        </View>
      )}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    />
  );
};

export default ResultsScreen;
