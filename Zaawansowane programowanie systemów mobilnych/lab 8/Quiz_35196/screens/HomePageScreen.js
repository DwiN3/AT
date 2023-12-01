// HomePageScreen.js

import React from 'react';
import { View, Text, TouchableOpacity, FlatList } from 'react-native';
import { TestsList } from '../data/Tests';
import styles from '../styles/HomePageStyle';

const truncateText = (text, maxLength) => {
  if (text.length > maxLength) {
    return text.substring(0, maxLength - 3) + '...';
  } else {
    return text;
  }
};

const HomePageScreen = ({ navigation }) => {
  const TestsListWithNewItem = [...TestsList, { resultsItem: true }];

  const renderResultsItem = ({ item }) => {
    if (item.resultsItem) {
      return (
        <View style={styles.resultsItem}>
          <Text style={styles.resultsItemText}>Get to know your ranking result</Text>
          <TouchableOpacity onPress={() => navigation.navigate('Results')}>
            <View style={styles.resultsItemButton}>
              <Text style={styles.resultsItemButtonText}>Check!</Text>
            </View>
          </TouchableOpacity>
        </View>
      );
    } else {
      return (
        <TouchableOpacity
          onPress={() =>
            navigation.navigate('Test', { test: item, titleTest: item.titleTest, tasks: item.tasks, description: item.description })
          }
        >
          <View style={[styles.testItem, styles.regularItem]}>
            <Text style={styles.titleTest}>{item.titleTest}</Text>
            <View style={styles.tagsContainer}>
              {item.tags.map((tag, index) => (
                <TouchableOpacity key={index} onPress={() => console.log(`Pressed ${tag}`)}>
                  <Text style={styles.tag}>{tag}</Text>
                </TouchableOpacity>
              ))}
            </View>
            <Text style={styles.description}>{truncateText(item.description, 100)}</Text>
          </View>
        </TouchableOpacity>
      );
    }
  };

  return (
    <View style={styles.containerHome}>
      <FlatList
        data={TestsListWithNewItem}
        renderItem={renderResultsItem}
        keyExtractor={(item) => item.titleTest || 'resultsItem'}
        contentContainerStyle={styles.flatListContainer}
      />
    </View>
  );
};

export default HomePageScreen;