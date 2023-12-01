// ResultsScreen.js

import React from 'react';
import { ScrollView, View, Text } from 'react-native';
import styles from '../styles/ResultsStyle';
import { userScores } from '../data/UserScores';

const ResultsScreen = () => (
  <ScrollView contentContainerStyle={styles.scrollContainer}>
    <View style={[styles.row, styles.header]}>
      <Text style={styles.cell}>Nick</Text>
      <Text style={styles.cell}>Point</Text>
      <Text style={styles.cell}>Type</Text>
      <Text style={styles.cell}>Date</Text>
    </View>
    {userScores.map((user, index) => (
      <View key={index} style={[styles.row, index % 2 === 0 ? styles.evenRow : styles.oddRow]}>
        <Text style={styles.cell}>{user.nick}</Text>
        <Text style={styles.cell}>{user.point}</Text>
        <Text style={styles.cell}>{user.type}</Text>
        <Text style={styles.cell}>{user.date}</Text>
      </View>
    ))}
  </ScrollView>
);

export default ResultsScreen;