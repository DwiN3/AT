// HomePageScreen.js

import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, FlatList } from 'react-native';
import styles from '../styles/HomePageStyle';

const truncateText = (text, maxLength) => {
  if (text.length > maxLength) {
    return text.substring(0, maxLength - 3) + '...';
  } else {
    return text;
  }
};

const HomePageScreen = ({ navigation }) => {
  const [testsList, setTestsList] = useState([]);

  useEffect(() => {
    const fetchTests = async () => {
      try {
        const response = await fetch('https://tgryl.pl/quiz/tests');
        const data = await response.json();
        setTestsList([...data, { resultsItem: true }]);
      } catch (error) {
        console.error('Error fetching tests:', error);
      }
    };

    fetchTests();
  }, []);

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
      const maxDescriptionLength = item.name.length > 30 ? 50 : 100;
      return (
        <TouchableOpacity
          onPress={() =>
            navigation.navigate('Test', {
              testId: item.id,
              titleTest: item.name,
              typeTest: item.type,
            })
          }
        >
          <View style={[styles.testItem, styles.regularItem]}>
            <Text style={styles.titleTest}>{item.name}</Text>
            <View style={styles.tagsContainer}>
              {item.tags.map((tag, index) => (
                <TouchableOpacity key={index} onPress={() => console.log(`Pressed ${tag}`)}>
                  <Text style={styles.tag}>{tag}</Text>
                </TouchableOpacity>
              ))}
            </View>
            <Text style={styles.description}>{truncateText(item.description, maxDescriptionLength)}</Text>
          </View>
        </TouchableOpacity>
      );
    }
  };

  return (
    <View style={styles.containerHome}>
      <FlatList
        data={testsList}
        renderItem={renderResultsItem}
        keyExtractor={(item) => item.name || 'resultsItem'}
        contentContainerStyle={styles.flatListContainer}
      />
    </View>
  );
};

export default HomePageScreen;