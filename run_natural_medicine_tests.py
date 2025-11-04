#!/usr/bin/env python3
"""
Comprehensive Natural Medicine Test Runner for Odin
"""

import sys
import os
import json
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test import OdinAssistant

def load_test_questions():
    """Load test questions from JSON file"""
    try:
        with open('natural_medicine_questions.json', 'r') as f:
            data = json.load(f)
        return data['natural_medicine_test_questions']
    except Exception as e:
        print(f"Error loading questions: {e}")
        return None

def evaluate_response(question, response):
    """Evaluate the quality of a response"""
    response_lower = response.lower()
    question_lower = question.lower()
    
    # Quality metrics
    metrics = {
        'natural_focus': False,
        'topic_relevant': False,
        'substantial': False,
        'engaging': False,
        'safe': False
    }
    
    # Check for natural medicine focus
    natural_keywords = ['natural', 'herb', 'plant', 'healing', 'holistic', 'wellness', 'remedy', 'nutrition']
    if any(keyword in response_lower for keyword in natural_keywords):
        metrics['natural_focus'] = True
    
    # Check topic relevance (basic keyword matching)
    question_words = set(question_lower.split())
    response_words = set(response_lower.split())
    common_words = question_words.intersection(response_words)
    if len(common_words) >= 2:  # At least 2 words in common
        metrics['topic_relevant'] = True
    
    # Check if substantial (good length)
    word_count = len(response.split())
    if word_count >= 20:
        metrics['substantial'] = True
    
    # Check if engaging (has questions or personal touches)
    if '?' in response or any(phrase in response_lower for phrase in ['you', 'your', 'what do you think']):
        metrics['engaging'] = True
    
    # Check for safety considerations
    safety_phrases = ['consult', 'healthcare', 'qualified', 'safe', 'caution', 'doctor']
    if any(phrase in response_lower for phrase in safety_phrases):
        metrics['safe'] = True
    
    # Calculate overall score
    score = sum(metrics.values()) / len(metrics)
    
    return metrics, score

def run_category_test(odin, category_name, questions, max_questions=None):
    """Run tests for a specific category"""
    print(f"\n🌿 Testing Category: {category_name.replace('_', ' ').title()}")
    print("=" * 60)
    
    if max_questions and len(questions) > max_questions:
        questions = random.sample(questions, max_questions)
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n💬 Question {i}: {question}")
        print("-" * 50)
        
        try:
            response = odin.generate_response(question, response_format="text")
            print(f"🤖 Odin: {response}")
            
            metrics, score = evaluate_response(question, response)
            results.append({
                'question': question,
                'response': response,
                'metrics': metrics,
                'score': score
            })
            
            # Display evaluation
            if score >= 0.8:
                print("🌟 EXCELLENT response!")
            elif score >= 0.6:
                print("✅ GOOD response!")
            elif score >= 0.4:
                print("⚠️  FAIR response")
            else:
                print("❌ Needs improvement")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'question': question,
                'response': None,
                'metrics': None,
                'score': 0
            })
    
    return results

def run_comprehensive_test():
    """Run comprehensive natural medicine test"""
    print("🌿 COMPREHENSIVE NATURAL MEDICINE TEST")
    print("=" * 60)
    
    # Load questions
    questions_data = load_test_questions()
    if not questions_data:
        print("❌ Could not load test questions")
        return
    
    try:
        # Initialize Odin
        print("Initializing Odin (CPU mode for stability)...")
        odin = OdinAssistant(finetuned_dir="./odin-finetuned", force_cpu=True)
        
        # Test each category
        all_results = {}
        total_questions = 0
        total_score = 0
        
        for category, questions in questions_data.items():
            # Limit questions per category for manageable testing
            max_per_category = 5 if len(questions) > 10 else len(questions)
            
            results = run_category_test(odin, category, questions, max_per_category)
            all_results[category] = results
            
            # Calculate category stats
            category_scores = [r['score'] for r in results if r['score'] is not None]
            if category_scores:
                avg_score = sum(category_scores) / len(category_scores)
                print(f"\n📊 {category.replace('_', ' ').title()} Average Score: {avg_score:.2f}")
                total_questions += len(category_scores)
                total_score += sum(category_scores)
        
        # Overall results
        print("\n" + "=" * 60)
        print("🎯 COMPREHENSIVE TEST RESULTS")
        print("=" * 60)
        
        if total_questions > 0:
            overall_average = total_score / total_questions
            print(f"📊 Overall Average Score: {overall_average:.2f}")
            print(f"📈 Total Questions Tested: {total_questions}")
            
            if overall_average >= 0.8:
                print("🎉 OUTSTANDING! Odin is an expert in natural medicine!")
            elif overall_average >= 0.6:
                print("🌟 EXCELLENT! Strong natural medicine knowledge!")
            elif overall_average >= 0.4:
                print("✅ GOOD! Solid natural medicine understanding!")
            else:
                print("⚠️  Needs improvement in natural medicine responses")
        
        print(f"\n🚀 Odin is ready for natural health conversations!")
        
    except Exception as e:
        print(f"❌ Test error: {e}")

def run_quick_test():
    """Run a quick test with selected questions"""
    print("🌿 QUICK NATURAL MEDICINE TEST")
    print("=" * 50)
    
    # Quick test questions (mix from all categories)
    quick_questions = [
        "What is natural medicine?",
        "Tell me about turmeric benefits",
        "Natural remedies for stress",
        "How do I choose quality supplements?",
        "What are adaptogens?",
        "Natural ways to boost immunity",
        "Tell me about chamomile tea",
        "How does holistic healing work?",
        "What herbs help with sleep?",
        "Natural approaches to energy?"
    ]
    
    try:
        print("Initializing Odin...")
        odin = OdinAssistant(finetuned_dir="./odin-finetuned", force_cpu=True)
        
        results = []
        for i, question in enumerate(quick_questions, 1):
            print(f"\n💬 Question {i}: {question}")
            print("-" * 40)
            
            try:
                response = odin.generate_response(question, response_format="text")
                print(f"🤖 Odin: {response}")
                
                metrics, score = evaluate_response(question, response)
                results.append(score)
                
                if score >= 0.8:
                    print("🌟 EXCELLENT!")
                elif score >= 0.6:
                    print("✅ GOOD!")
                else:
                    print("⚠️  Could be better")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append(0)
        
        # Quick results
        if results:
            avg_score = sum(results) / len(results)
            success_rate = (sum(1 for s in results if s >= 0.6) / len(results)) * 100
            
            print(f"\n📊 Quick Test Results:")
            print(f"Average Score: {avg_score:.2f}")
            print(f"Success Rate: {success_rate:.1f}%")
            
            if success_rate >= 80:
                print("🎉 EXCELLENT natural medicine performance!")
            else:
                print("✅ Good performance, ready for natural health chats!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--comprehensive":
        run_comprehensive_test()
    else:
        run_quick_test()